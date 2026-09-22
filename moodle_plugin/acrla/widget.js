(function() {
    'use strict';

    /*
     * ACRLA Moodle widget controller.
     *
     * Purpose:
     *   Runs inside Moodle pages and connects Moodle context to the ACRLA
     *   FastAPI/frontend iframe.
     *
     * Main responsibilities:
     *   - render dashboard/course/chapter mastery controls
     *   - open one persistent right-side ACRLA panel
     *   - launch overview, course remediation, or chapter remediation
     *   - sync Moodle profile/material/mastery data with the backend
     *   - refresh mastery buttons after assessment updates
     */

    function initAcrlaWidget() {
        var root = document.getElementById('acrla-widget');
        if (!root || root.dataset.initialized === '1') {
            return;
        }
        root.dataset.initialized = '1';

        var launcher = document.getElementById('acrla-launcher');
        var remediationLink = document.getElementById('acrla-remediation-link');
        var materialSync = document.getElementById('acrla-material-sync');
        var gradeLinks = Array.prototype.slice.call(root.querySelectorAll('.acrla-grade-link'));
        var chapterGradeTemplates = gradeLinks.filter(function(link) {
            return link.getAttribute('data-acrla-level-type') === 'chapter';
        });
        var panel = document.getElementById('acrla-panel');
        var close = document.getElementById('acrla-close');
        var frame = document.getElementById('acrla-frame');
        var status = document.getElementById('acrla-sync-status');
        var panelTitle = panel ? panel.querySelector('#acrla-panel-header strong') : null;
        var panelSubtitle = panel ? panel.querySelector('#acrla-panel-header span:not(#acrla-sync-status)') : null;
        var backendUrl = (root.dataset.backendUrl || '').replace(/\/+$/, '');
        var iframeUrl = root.dataset.iframeUrl || backendUrl;
        var materialSyncUrl = root.dataset.materialSyncUrl || '';
        var syncPayload = parsePayload(root.dataset.syncPayload || '{}');
        var dashboardSyncPayloads = parsePayload(root.dataset.dashboardSyncPayloads || '[]');
        var synced = false;
        var dashboardSynced = false;
        var pendingMasteryUpdate = null;
        var latestMasteryData = null;
        var activeCourseId = String(syncPayload.course_id || '');

        if (!launcher || !panel || !close || !frame || !backendUrl) {
            return;
        }

        launcher.addEventListener('click', function() {
            // The floating ACRLA launcher is the overview entry point. Grade
            // and mastery buttons below use scoped remediation launch paths.
            var willOpen = !panel.classList.contains('acrla-open');
            if (willOpen) {
                openPersistentAcrlaPanel('overview', {});
            } else {
                setOpen(false);
            }
        });

        if (remediationLink) {
            remediationLink.style.display = 'none';
            remediationLink.setAttribute('aria-hidden', 'true');
            remediationLink.addEventListener('click', function(event) {
                event.preventDefault();
                event.stopPropagation();
                var launchUrl = remediationLink.getAttribute('data-launch-url') || remediationLink.getAttribute('href');
                if (launchUrl) {
                    activeCourseId = remediationLink.getAttribute('data-acrla-course-id') || activeCourseId;
                    openGradeFrame(launchUrl);
                } else {
                    setOpen(true);
                    setStatus('Remediation link is missing launch context.');
                }
            });
        }

        if (materialSync) {
            materialSync.addEventListener('click', function(event) {
                event.preventDefault();
                event.stopPropagation();
                syncCourseMaterials();
            });
        }

        if (root.dataset.acrlaDashboard === '1') {
            gradeLinks.forEach(bindGradeLink);
            hideDashboardTemplatePanel();
            renderDashboardMastery();
            window.setTimeout(renderDashboardMastery, 500);
            window.setTimeout(renderDashboardMastery, 1500);
            window.setTimeout(renderDashboardMastery, 3000);
        } else {
            removeChapterRemediationSidebar();
        }
        // Refresh on all pages (not just dashboard) so course-page buttons
        // show ACRLA mastery on load instead of the PHP-rendered Moodle grade.
        renderInlineChapterRemediation();
        window.setTimeout(renderInlineChapterRemediation, 700);
        startInlineRemediationObserver();
        refreshMasteryDisplay();
        window.setTimeout(refreshMasteryDisplay, 1200);
        refreshMasteryFromBackend();
        window.setTimeout(refreshMasteryFromBackend, 1500);

        window.addEventListener('message', function(event) {
            var data = event.data || {};
            if (data.type === 'acrla:mastery-updated') {
                if (window.console && window.console.log) {
                    window.console.log('[ACRLA] mastery update event', JSON.stringify(data));
                }
                pendingMasteryUpdate = data;
                applyMasteryUpdateEvent(data);
                forceMasteryRefresh('assessment-submit');
            }
        });

        function bindGradeLink(link) {
            link.addEventListener('click', function(event) {
                return openAcrlaFromGradeElement(link, event);
            });
            link.addEventListener('keydown', function(event) {
                if (event.key !== 'Enter' && event.key !== ' ') {
                    return;
                }
                return openAcrlaFromGradeElement(link, event);
            });
        }

        // ==========================================================
        // Launch Event Handlers
        // ==========================================================

        function openAcrlaFromGradeElement(link, event) {
            /*
             * Shared handler for overall/course/chapter mastery controls. The
             * stable data attributes rendered by PHP/JS determine the launch
             * level, course, concept, and score sent to ACRLA.
             */
            if (event) {
                event.preventDefault();
                event.stopPropagation();
                if (event.stopImmediatePropagation) {
                    event.stopImmediatePropagation();
                }
            }
            if ((link.getAttribute('data-acrla-level-type') || '') === 'chapter') {
                openAcrlaChatPanelForChapter(link);
                return false;
            }
            var launchUrl = buildGradeLaunchUrl(link);
            var levelType = link.getAttribute('data-acrla-level-type') || 'course';
            activeCourseId = String(link.getAttribute('data-acrla-course-id') || activeCourseId || '');
            if (window.console && window.console.log) {
                window.console.log(
                    '[ACRLA] widget_grade_click',
                    'level_type=' + (link.getAttribute('data-acrla-level-type') || 'chapter'),
                    'course_id=' + (link.getAttribute('data-acrla-course-id') || ''),
                    'course_name=' + (link.getAttribute('data-acrla-course-name') || ''),
                    'concept=' + (link.getAttribute('data-acrla-concept') || ''),
                    'launch_url=' + launchUrl
                );
            }
            if (launchUrl) {
                openPersistentAcrlaPanel(levelType === 'course' ? 'course_chat' : 'overview', {
                    triggerEl: link,
                    launchUrl: launchUrl
                });
            } else {
                setOpen(true);
                setStatus('Grade launch context is missing.');
            }
            return false;
        }

        function openGradeElement(link, event) {
            return openAcrlaFromGradeElement(link, event);
        }

        close.addEventListener('click', function() {
            setOpen(false);
        });

        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                setOpen(false);
            }
        });

        function setOpen(open) {
            panel.classList.toggle('acrla-open', open);
            panel.setAttribute('aria-hidden', open ? 'false' : 'true');
            launcher.setAttribute('aria-expanded', open ? 'true' : 'false');
        }

        function setPanelHeading(title, subtitle) {
            if (panelTitle) {
                panelTitle.textContent = title;
            }
            if (panelSubtitle) {
                panelSubtitle.textContent = subtitle;
            }
        }

        function openPersistentAcrlaPanel(mode, options) {
            /*
             * Reuse the same Moodle side panel for every trigger. The bottom
             * ACRLA button opens the overview; course/chapter controls open
             * scoped remediation without creating nested ACRLA windows.
             *
             * The iframe receives launch query parameters, but the backend
             * still validates and rebuilds the real remediation scope.
             */
            options = options || {};
            panel.classList.remove('acrla-panel-overview-only');
            if (mode === 'chapter_chat') {
                var triggerEl = options.triggerEl;
                if (!triggerEl) {
                    setOpen(true);
                    setStatus('Chapter launch context is missing.');
                    return;
                }
                var concept = options.concept || triggerEl.getAttribute('data-acrla-concept') || '';
                var score = options.score || triggerEl.getAttribute('data-acrla-score') || '';
                var chapterCourseName = options.courseName || triggerEl.getAttribute('data-acrla-course-name') || syncPayload.course_name || '';
                var chapterTitle = cleanChapterTitle(concept);
                var chapterUrl = addQueryParams(options.launchUrl || buildGradeLaunchUrl(triggerEl), {
                    embedded: '1',
                    hide_header: '1',
                    concept: concept,
                    course_name: chapterCourseName,
                    score: score,
                    status: options.status || getMasteryStatus(score).toLowerCase()
                });
                activeCourseId = String(options.courseId || triggerEl.getAttribute('data-acrla-course-id') || activeCourseId || '');
                panel.classList.add('acrla-chat-direct');
                setPanelHeading('ACRLA', 'Chapter remediation: ' + chapterTitle + (chapterCourseName ? ' · Course: ' + chapterCourseName : ''));
                setOpen(true);
                setStatus('Launching ' + chapterTitle + ' remediation...');
                if (window.console && window.console.log) {
                    window.console.log(
                        '[ACRLA] persistent_panel_open',
                        'mode=chapter_chat',
                        'course_id=' + (triggerEl.getAttribute('data-acrla-course-id') || ''),
                        'concept=' + concept,
                        'score=' + score,
                        'launch_url=' + chapterUrl
                    );
                }
                if (chapterUrl) {
                    frame.setAttribute('src', chapterUrl);
                    setStatus(chapterTitle + ' remediation opened');
                } else {
                    setStatus('Grade launch context is missing.');
                }
                return;
            }

            if (mode === 'overall_chat') {
                var overallScore = String(options.score || 0);
                var overallStatus = String(options.status || getMasteryStatus(Number(overallScore)).toLowerCase());
                var overallUrl = addQueryParams(iframeUrl, {
                    embedded: '1',
                    hide_header: '1',
                    level_type: 'overall',
                    score: overallScore,
                    status: overallStatus
                });
                panel.classList.add('acrla-chat-direct');
                setPanelHeading('ACRLA', 'Overall remediation');
                setOpen(true);
                setStatus('Launching overall remediation...');
                if (window.console && window.console.log) {
                    window.console.log(
                        '[ACRLA] persistent_panel_open',
                        'mode=overall_chat',
                        'score=' + overallScore,
                        'status=' + overallStatus,
                        'launch_url=' + overallUrl
                    );
                }
                frame.setAttribute('src', overallUrl);
                setStatus('Overall remediation opened');
                return;
            }

            if (mode === 'course_chat') {
                var courseEl = options.triggerEl;
                var courseName = options.courseName || (courseEl ? courseEl.getAttribute('data-acrla-course-name') : '') || syncPayload.course_name || 'this course';
                var courseUrl = addQueryParams(options.launchUrl || (courseEl ? buildGradeLaunchUrl(courseEl) : ''), {
                    embedded: '1',
                    hide_header: '1',
                    course_name: courseName
                });
                activeCourseId = String(options.courseId || (courseEl ? courseEl.getAttribute('data-acrla-course-id') : '') || activeCourseId || '');
                panel.classList.add('acrla-chat-direct');
                setPanelHeading('ACRLA', 'Course remediation: ' + courseName);
                setOpen(true);
                setStatus('Launching ' + courseName + ' remediation...');
                if (courseUrl) {
                    frame.setAttribute('src', courseUrl);
                    setStatus(courseName + ' remediation opened');
                } else {
                    setStatus('Course launch context is missing.');
                }
                return;
            }

            panel.classList.remove('acrla-chat-direct');
            panel.classList.add('acrla-panel-overview-only');
            setPanelHeading('ACRLA', 'Adaptive Conversation & Remediation');
            frame.setAttribute('src', 'about:blank');
            setOpen(true);
            setStatus('');
            forceMasteryRefresh('panel-open');
        }

        function openAcrlaOverviewPanel() {
            openPersistentAcrlaPanel('overview', {});
        }

        function openAcrlaChatPanelForChapter(triggerEl) {
            var concept = triggerEl.getAttribute('data-acrla-concept') || '';
            var score = triggerEl.getAttribute('data-acrla-score') || '';
            openPersistentAcrlaPanel('chapter_chat', {
                triggerEl: triggerEl,
                courseId: triggerEl.getAttribute('data-acrla-course-id') || '',
                concept: concept,
                score: score,
                courseName: triggerEl.getAttribute('data-acrla-course-name') || '',
                status: getMasteryStatus(score).toLowerCase()
            });
        }

        function openFrame(url, syncFirst) {
            setOpen(true);
            setStatus(syncFirst ? 'Opening ACRLA...' : 'Launching remediation...');
            var ready = syncFirst ? syncOnce() : Promise.resolve();
            runFinally(ready, function() {
                frame.setAttribute('src', url);
                setStatus(syncFirst ? 'ACRLA opened' : 'Remediation opened');
            });
        }

        function openGradeFrame(url) {
            setOpen(true);
            setStatus(root.dataset.acrlaDashboard === '1' ? 'Syncing course mastery...' : 'Launching remediation...');
            var ready = root.dataset.acrlaDashboard === '1' ? syncDashboardCourses() : Promise.resolve();
            runFinally(ready, function() {
                frame.setAttribute('src', url);
                setStatus('Remediation opened');
            });
        }

        function syncOnce() {
            if (synced) {
                return Promise.resolve();
            }
            synced = true;
            setStatus('Syncing Moodle data...');

            return fetch(backendUrl + '/api/v1/moodle/sync', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(syncPayload),
                credentials: 'omit'
            })
                .then(function(response) {
                    if (!response.ok) {
                        throw new Error('HTTP ' + response.status);
                    }
                    return response.json();
                })
                .then(function(data) {
                    if (data && data.status === 'ok') {
                        setStatus('Moodle data synced');
                        // Apply sync response mastery directly to course-page buttons.
                        // The sync response is keyed by the original PHP-cleaned concept
                        // names so normalized matching is used for safety.
                        if (data.mastery && syncPayload.course_id) {
                            applySyncMastery(String(syncPayload.course_id), data.mastery);
                        }
                    } else {
                        setStatus('Opened without sync confirmation');
                    }
                })
                .catch(function(error) {
                    synced = false;
                    setStatus('Sync failed');
                    if (window.console && window.console.warn) {
                        window.console.warn('ACRLA Moodle sync failed:', error);
                    }
                });
        }

        function syncDashboardCourses() {
            if (dashboardSynced || !Array.isArray(dashboardSyncPayloads) || dashboardSyncPayloads.length === 0) {
                return Promise.resolve();
            }
            dashboardSynced = true;
            setStatus('Syncing dashboard mastery...');

            return Promise.all(dashboardSyncPayloads.map(function(payload) {
                return fetch(backendUrl + '/api/v1/moodle/sync', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload),
                    credentials: 'omit'
                }).then(function(response) {
                    if (!response.ok) {
                        throw new Error('HTTP ' + response.status);
                    }
                    return response.json();
                });
            })).then(function() {
                setStatus('Dashboard mastery synced');
                refreshMasteryDisplay();
            }).catch(function(error) {
                dashboardSynced = false;
                setStatus('Dashboard sync failed');
                if (window.console && window.console.warn) {
                    window.console.warn('ACRLA dashboard sync failed:', error);
                }
            });
        }

        function syncCourseMaterials() {
            /*
             * Moodle owns the files. The plugin uploads PDFs to ACRLA through
             * the API instead of letting the backend query Moodle directly.
             */
            if (!materialSyncUrl) {
                setOpen(true);
                setStatus('Course material sync URL is missing.');
                return;
            }

            setOpen(true);
            setStatus('Syncing course PDFs...');
            materialSync.disabled = true;

            var request = fetch(materialSyncUrl, {
                method: 'POST',
                credentials: 'same-origin'
            })
                .then(function(response) {
                    if (!response.ok) {
                        throw new Error('HTTP ' + response.status);
                    }
                    return response.json();
                })
                .then(function(data) {
                    if (data && data.status === 'ok') {
                        setStatus('Synced ' + (data.files_synced || 0) + ' PDF file(s) to ACRLA');
                        window.setTimeout(refreshMasteryFromBackend, 500);
                    } else {
                        setStatus('Course material sync did not complete.');
                    }
                })
                .catch(function(error) {
                    setStatus('Course material sync failed.');
                    if (window.console && window.console.warn) {
                        window.console.warn('ACRLA course material sync failed:', error);
                    }
                });

            runFinally(request, function() {
                materialSync.disabled = false;
            });
        }

        function buildGradeLaunchUrl(link) {
            /*
             * Build the FastAPI `/moodle/launch` URL from the clicked Moodle
             * mastery control. This keeps all grade-click flows on the same
             * backend launch contract.
             */
            var concept = link.getAttribute('data-acrla-concept') || '';
            var score = link.getAttribute('data-acrla-score') || '';
            var levelType = link.getAttribute('data-acrla-level-type') || 'chapter';
            if (!score || (levelType === 'chapter' && !concept)) {
                return '';
            }

            var params = new URLSearchParams();
            params.set('student_id', link.getAttribute('data-acrla-student-id') || syncPayload.student_id || '');
            params.set('student_name', link.getAttribute('data-acrla-student-name') || syncPayload.student_name || '');
            if (levelType !== 'overall') {
                params.set('course_id', link.getAttribute('data-acrla-course-id') || syncPayload.course_id || '');
                params.set('course_name', link.getAttribute('data-acrla-course-name') || syncPayload.course_name || '');
            }
            if (concept) {
                params.set('concept', concept);
            }
            params.set('score', score);
            if (levelType === 'course') {
                params.set('course_score', score);
            }
            if (levelType === 'overall') {
                params.set('overall_score', score);
            }
            params.set('level_type', levelType);
            params.set('source', 'moodle');

            var launchUrl = backendUrl + '/api/v1/moodle/launch?' + params.toString();
            if (window.console && window.console.log) {
                window.console.log(
                    '[ACRLA] widget_launch_url',
                    'level_type=' + levelType,
                    'course_id=' + (params.get('course_id') || ''),
                    'course_name=' + (params.get('course_name') || ''),
                    'concept=' + (params.get('concept') || ''),
                    'score=' + (params.get('score') || ''),
                    'url=' + launchUrl
                );
            }
            return launchUrl;
        }

        function hideDashboardTemplatePanel() {
            var panel = root.querySelector('#acrla-grade-remediation');
            if (panel) {
                panel.style.display = 'none';
            }
        }

        function removeChapterRemediationSidebar() {
            var panel = root.querySelector('#acrla-grade-remediation');
            if (panel) {
                panel.remove();
            }
        }

        function renderDashboardMastery() {
            /*
             * Render dashboard mastery controls from discovered Moodle courses
             * and backend mastery data; avoid hardcoded course ids.
             */
            renderOverallMastery();
            renderCourseMasteryButtons();
            renderMasteryLegend();
        }

        function renderOverallMastery() {
            if (root.dataset.acrlaDashboard !== '1') {
                return;
            }
            if (document.getElementById('acrla-overall-mastery-card')) {
                return;
            }

            var overallButton = root.querySelector('.acrla-grade-link[data-acrla-level-type="overall"]');
            if (!overallButton) {
                return;
            }

            var score = Number(overallButton.getAttribute('data-acrla-score') || 0);
            var statusText = getMasteryStatus(score);
            var btn = document.createElement('button');
            btn.type = 'button';
            btn.id = 'acrla-overall-mastery-card';
            btn.className = 'acrla-overall-mastery-card ' + getMasteryClass(score);
            btn.setAttribute('data-acrla-level-type', 'overall');
            btn.setAttribute('data-acrla-score', String(score));
            btn.setAttribute('data-acrla-status', statusText.toLowerCase());
            btn.setAttribute('aria-label', 'Open ACRLA overall remediation');
            btn.innerHTML =
                '<span class="acrla-overall-title">Overall ACRLA mastery</span>' +
                '<span class="acrla-overall-score">' + formatScore(score) + '%</span>' +
                '<span class="acrla-overall-status"><span class="acrla-status-dot" aria-hidden="true"></span>' + statusText + '</span>';
            btn.addEventListener('click', function() {
                var currentScore = Number(btn.getAttribute('data-acrla-score') || 0);
                openPersistentAcrlaPanel('overall_chat', {
                    score: currentScore,
                    status: getMasteryStatus(currentScore).toLowerCase()
                });
            });

            var target = findCourseOverviewTarget();
            target.parentNode.insertBefore(btn, target);
            if (window.console && window.console.log) {
                window.console.log('[ACRLA overall] compact overall mastery button rendered');
            }
        }

        function renderCourseMasteryButtons() {
            if (window.console && window.console.log) {
                window.console.log('[ACRLA] renderCourseMasteryButtons CALLED');
            }
            if (root.dataset.acrlaDashboard !== '1') {
                return;
            }
            logDomDiagnostics();

            var courseButtons = dashboardCourseButtonMap();
            var inserted = {};
            var dataCourseEntries = courseEntriesFromDataAttributes();
            var links = Array.prototype.filter.call(
                document.querySelectorAll('a[href*="/course/view.php?id="], a[href*="course/view.php?id="]'),
                function(link) {
                    return !root.contains(link) && extractCourseId(link.getAttribute('href'));
                }
            );
            var cards = dataCourseEntries.slice();
            var seenCards = dataCourseEntries.map(function(entry) {
                return entry.card;
            });
            Array.prototype.forEach.call(links, function(link) {
                var card = findCourseCard(link);
                if (card && seenCards.indexOf(card) === -1) {
                    seenCards.push(card);
                    cards.push({
                        card: card,
                        link: link,
                        courseId: extractCourseId(link.getAttribute('href')),
                        courseName: link.textContent.trim()
                    });
                }
            });
            var nameMatchedCards = [];
            Object.keys(courseButtons).forEach(function(courseId) {
                var courseName = courseButtons[courseId].getAttribute('data-acrla-course-name') || '';
                var card = findCourseCardByName(courseName);
                if (card && nameMatchedCards.indexOf(card) === -1) {
                    nameMatchedCards.push(card);
                }
            });
            if (window.console && window.console.log) {
                window.console.log('[ACRLA] course links found', links.length);
                window.console.log('[ACRLA] found course cards:', Math.max(cards.length, nameMatchedCards.length));
            }

            Array.prototype.forEach.call(cards, function(entry) {
                var courseLink = entry.link;
                var courseId = entry.courseId;
                if (!courseId || inserted[courseId]) {
                    return;
                }
                var button = courseButtons[courseId];
                if (!button) {
                    return;
                }
                var card = entry.card;
                if (!card || card.querySelector('.acrla-course-card-button[data-acrla-course-id="' + cssEscape(courseId) + '"]')) {
                    return;
                }
                var clone = button.cloneNode(true);
                clone.classList.remove('acrla-dashboard-course-button');
                clone.classList.add('acrla-course-card-button');
                clone.setAttribute('aria-label', 'Open ACRLA course remediation');
                updateMasteryButtonVisual(clone, Number(clone.getAttribute('data-acrla-score') || 0));
                bindGradeLink(clone);
                if (window.console && window.console.log) {
                    window.console.log('[ACRLA] rendering course button:', courseId, clone.getAttribute('data-acrla-course-name') || entry.courseName || (courseLink ? courseLink.textContent.trim() : ''));
                }
                insertCourseCardButton(card, clone);
                inserted[courseId] = true;
            });

            Object.keys(courseButtons).forEach(function(courseId) {
                if (inserted[courseId]) {
                    return;
                }
                var button = courseButtons[courseId];
                var courseName = button.getAttribute('data-acrla-course-name') || '';
                var card = findCourseCardByName(courseName);
                if (!card || card.querySelector('.acrla-course-card-button[data-acrla-course-id="' + cssEscape(courseId) + '"]')) {
                    return;
                }
                var clone = button.cloneNode(true);
                clone.classList.remove('acrla-dashboard-course-button');
                clone.classList.add('acrla-course-card-button');
                clone.setAttribute('aria-label', 'Open ACRLA course remediation');
                updateMasteryButtonVisual(clone, Number(clone.getAttribute('data-acrla-score') || 0));
                bindGradeLink(clone);
                if (window.console && window.console.log) {
                    window.console.log('[ACRLA] rendering course button:', courseId, courseName);
                }
                insertCourseCardButton(card, clone);
                inserted[courseId] = true;
            });
        }

        function refreshMasteryDisplay() {
            var studentId = syncPayload.student_id || (dashboardSyncPayloads[0] && dashboardSyncPayloads[0].student_id);
            if (!studentId || !backendUrl) {
                return Promise.resolve();
            }
            return fetch(backendUrl + '/api/v1/mastery/student/' + encodeURIComponent(studentId), {
                method: 'GET',
                credentials: 'omit'
            })
                .then(function(response) {
                    if (!response.ok) {
                        throw new Error('HTTP ' + response.status);
                    }
                    return response.json();
                })
                .then(function(data) {
                    applyMasteryData(data);
                })
                .catch(function(error) {
                    if (window.console && window.console.warn) {
                        window.console.warn('ACRLA mastery refresh failed:', error);
                    }
                });
        }

        function refreshMasteryFromBackend() {
            /*
             * Canonical UI refresh after assessment updates. Buttons should
             * display current ACRLA mastery, not stale Moodle baseline scores.
             */
            var studentId = syncPayload.student_id || (Array.isArray(dashboardSyncPayloads) && dashboardSyncPayloads[0] && dashboardSyncPayloads[0].student_id);
            if (!studentId || !backendUrl) {
                return Promise.resolve();
            }
            return fetch(backendUrl + '/api/v1/mastery/student/' + encodeURIComponent(studentId), {
                method: 'GET',
                credentials: 'omit'
            })
            .then(function(r) { return r.ok ? r.json() : null; })
            .then(function(data) {
                if (!data || data.status !== 'ok') {
                    return;
                }
                applyMasteryData(data);
            })
            .catch(function(err) {
                if (window.console && window.console.warn) {
                    window.console.warn('[ACRLA] refreshMasteryFromBackend failed:', err);
                }
            });
        }

        function forceMasteryRefresh(reason) {
            if (window.console && window.console.log) {
                window.console.log('[ACRLA] force mastery refresh', reason || '');
            }
            refreshMasteryDisplay();
            refreshMasteryFromBackend();
            window.setTimeout(refreshMasteryDisplay, 500);
            window.setTimeout(refreshMasteryFromBackend, 900);
            window.setTimeout(refreshMasteryDisplay, 1800);
        }

        function applyMasteryUpdateEvent(data) {
            var score = Number(data.current_acrla_mastery || data.updated_mastery || 0);
            if (!Number.isFinite(score)) {
                return;
            }
            var levelType = String(data.level_type || '').toLowerCase();
            var courseId = String(data.course_id || '');
            var courseSelector = '[data-acrla-level-type="course"][data-acrla-course-id="' + cssEscape(courseId) + '"]';
            var chapterSelector = '[data-acrla-level-type="chapter"][data-acrla-course-id="' + cssEscape(courseId) + '"]';
            if (levelType === 'overall') {
                Array.prototype.forEach.call(document.querySelectorAll('[data-acrla-level-type="overall"]'), function(button) {
                    updateGradeButton(button, score);
                });
                return;
            }
            if (levelType === 'course' && courseId) {
                var courseButtons = document.querySelectorAll(courseSelector);
                if (window.console && window.console.log) {
                    window.console.log('[ACRLA] event course update course_id=' + courseId + ' buttons=' + courseButtons.length + ' score=' + score);
                }
                Array.prototype.forEach.call(courseButtons, function(button) {
                    updateGradeButton(button, score);
                });
                return;
            }
            if (levelType === 'chapter' && courseId) {
                var concepts = String(data.concept || '').split(',').map(function(value) {
                    return normalizeConcept(value);
                }).filter(Boolean);
                var chapterButtons = document.querySelectorAll(chapterSelector);
                if (window.console && window.console.log) {
                    window.console.log('[ACRLA] event chapter update course_id=' + courseId + ' concepts=' + concepts.join('|') + ' buttons=' + chapterButtons.length + ' score=' + score);
                }
                Array.prototype.forEach.call(chapterButtons, function(button) {
                    var btnNorm = normalizeConcept(button.getAttribute('data-acrla-concept') || '');
                    var matched = concepts.some(function(conceptNorm) {
                        return conceptsMatch(btnNorm, conceptNorm);
                    });
                    if (matched) {
                        updateGradeButton(button, score);
                    }
                });
            }
        }

        function applyMasteryData(data) {
            if (window.console && window.console.log) {
                window.console.log('[ACRLA] mastery payload:', JSON.stringify(data));
            }
            if (!data || data.status !== 'ok') {
                return;
            }
            latestMasteryData = data;
            var overall = Number(data.overall_current_acrla_mastery || 0);
            Array.prototype.forEach.call(document.querySelectorAll('[data-acrla-level-type="overall"]'), function(button) {
                updateGradeButton(button, overall);
            });

            var courses = data.courses || [];
            courses.forEach(function(course) {
                var courseId = String(course.course_id || '');
                var score = Number(course.course_current_acrla_mastery || 0);
                var courseButtons = document.querySelectorAll('[data-acrla-level-type="course"][data-acrla-course-id="' + cssEscape(courseId) + '"]');
                if (window.console && window.console.log) {
                    window.console.log('[ACRLA] canonical course update course_id=' + courseId + ' buttons=' + courseButtons.length + ' current_acrla_mastery=' + score);
                }
                Array.prototype.forEach.call(courseButtons, function(button) {
                    updateGradeButton(button, score);
                });

                var concepts = course.concept_current_acrla_mastery || {};
                var conceptList = [];
                Object.keys(concepts).forEach(function(key) {
                    conceptList.push({
                        norm: normalizeConcept(key),
                        val: Number(concepts[key].current_acrla_mastery || 0),
                        orig: key,
                        source: concepts[key].source_of_truth || 'current_acrla_mastery'
                    });
                });
                Array.prototype.forEach.call(
                    document.querySelectorAll('[data-acrla-level-type="chapter"][data-acrla-course-id="' + cssEscape(courseId) + '"]'),
                    function(button) {
                        var btnConcept = button.getAttribute('data-acrla-concept') || '';
                        var btnNorm = normalizeConcept(btnConcept);
                        var match = null;
                        for (var i = 0; i < conceptList.length; i++) {
                            if (conceptsMatch(btnNorm, conceptList[i].norm)) {
                                match = conceptList[i];
                                break;
                            }
                        }
                        if (match !== null) {
                            if (window.console && window.console.log) {
                                window.console.log(
                                    '[ACRLA] chapter_mastery_display ' +
                                    'button_concept=' + btnConcept + ' ' +
                                    'normalized_concept_key=' + btnNorm + ' ' +
                                    'matched_backend_key=' + match.orig + ' ' +
                                    'concept_current_acrla_mastery=' + match.val + ' ' +
                                    'source_of_truth=' + match.source + ' ' +
                                    'course_current_acrla_mastery=' + score + ' ' +
                                    'value_written_strong=' + (Math.round(match.val * 10) / 10) + '% ' +
                                    'value_written_data_acrla_score=' + (Math.round(match.val * 10) / 10)
                                );
                            }
                            updateGradeButton(button, match.val);
                        } else {
                            if (window.console && window.console.log) {
                                window.console.log(
                                    '[ACRLA] chapter_mastery_display ' +
                                    'button_concept=' + btnConcept + ' ' +
                                    'normalized_concept_key=' + btnNorm + ' ' +
                                    'concept_current_acrla_mastery=NO_MATCH ' +
                                    'course_current_acrla_mastery=' + score + ' ' +
                                    'value_written_strong=unchanged ' +
                                    'value_written_data_acrla_score=unchanged ' +
                                    'available_concept_keys=' + conceptList.map(function(e) { return e.orig; }).join('|')
                                );
                            }
                        }
                    }
                );
            });
            if (pendingMasteryUpdate) {
                applyMasteryUpdateEvent(pendingMasteryUpdate);
            }
            renderPanelMasteryOverview(data);
            renderInlineChapterRemediation();
        }

        function updateGradeButton(button, score) {
            if (!button || Number.isNaN(score)) {
                return;
            }
            var rounded = Math.round(score * 100) / 100;
            var display = rounded.toFixed(2).replace(/\.?0+$/, '');
            button.setAttribute('data-acrla-score', String(rounded));
            var strong = button.querySelector('strong');
            if (strong) {
                strong.textContent = display + '%';
            }
            updateMasteryButtonVisual(button, rounded);
            if (latestMasteryData) {
                renderPanelMasteryOverview(latestMasteryData);
            }
        }

        function updateMasteryButtonVisual(button, score) {
            if (!button || !Number.isFinite(score)) {
                return;
            }
            var labelNode = button.querySelector('[data-acrla-title], span');
            var storedLabel = button.getAttribute('data-acrla-label') || '';
            var label = storedLabel || (labelNode ? labelNode.textContent.trim() : '') || 'ACRLA mastery';
            button.setAttribute('data-acrla-label', label);
            var levelType = button.getAttribute('data-acrla-level-type') || 'chapter';
            var courseName = button.getAttribute('data-acrla-course-name') || label;
            var concept = button.getAttribute('data-acrla-concept') || label;
            var status = getMasteryStatus(score);
            var masteryClass = getMasteryClass(score);
            setMasteryClass(button, score);

            if (button.classList.contains('acrla-inline-chapter-open')) {
                updateInlineRemediationVisual(button, score);
                return;
            }

            if (button.classList.contains('acrla-course-card-button')) {
                button.innerHTML =
                    '<span class="acrla-card-label">ACRLA course mastery</span>' +
                    '<strong class="acrla-card-score acrla-score-value">' + formatScore(score) + '%</strong>' +
                    renderMasteryBadge(score) +
                    '<span class="acrla-card-action acrla-course-card-assistant-button" title="Open ACRLA assistant" aria-label="Open ACRLA assistant"><span class="acrla-bot-icon" aria-hidden="true">&#129302;</span><span>Open ACRLA assistant</span></span>';
                return;
            }

            if (levelType === 'overall') {
                var overallStatus = getMasteryStatus(score);
                button.innerHTML =
                    '<span class="acrla-overall-title">Overall ACRLA mastery</span>' +
                    '<span class="acrla-overall-score">' + formatScore(score) + '%</span>' +
                    '<span class="acrla-overall-status"><span class="acrla-status-dot" aria-hidden="true"></span>' + overallStatus + '</span>';
                button.setAttribute('data-acrla-status', overallStatus.toLowerCase());
                return;
            }

            if (levelType === 'course') {
                button.innerHTML =
                    '<span class="acrla-row-icon" aria-hidden="true">' + courseIcon(courseName) + '</span>' +
                    '<span class="acrla-row-title" data-acrla-title>' + escapeHtml(courseName) + '</span>' +
                    '<strong class="acrla-score-value">' + formatScore(score) + '%</strong>' +
                    renderMasteryBadge(score);
                return;
            }

            button.innerHTML =
                '<span class="acrla-row-icon acrla-row-icon-chapter" aria-hidden="true">&#128214;</span>' +
                '<span class="acrla-row-title" data-acrla-title>' + escapeHtml(concept || label) + '</span>' +
                '<strong class="acrla-score-value">' + formatScore(score) + '%</strong>' +
                renderMasteryBadge(score);
        }

        function getMasteryStatus(score) {
            score = Number(score || 0);
            if (score < 50) {
                return 'Weak';
            }
            if (score < 80) {
                return 'Moderate';
            }
            return 'Strong';
        }

        function getMasteryClass(score) {
            score = Number(score || 0);
            if (score < 50) {
                return 'acrla-mastery-weak';
            }
            if (score < 80) {
                return 'acrla-mastery-moderate';
            }
            return 'acrla-mastery-strong';
        }

        function setMasteryClass(element, score) {
            if (!element) {
                return;
            }
            element.classList.remove('acrla-mastery-weak', 'acrla-mastery-moderate', 'acrla-mastery-strong');
            element.classList.add(getMasteryClass(score));
        }

        function renderInlineChapterRemediation() {
            /*
             * Add inline chapter mastery controls beside Moodle resources.
             * The score+badge opens scoped chapter chat; the PDF title remains
             * the normal Moodle resource link.
             */
            if (root.dataset.acrlaDashboard === '1') {
                return;
            }
            var chapters = chapterRemediationSources();
            if (!chapters.length) {
                return;
            }
            var targets = findMoodleChapterTargets();
            targets.forEach(function(target) {
                if (!target || target.row.querySelector('.acrla-inline-remediation')) {
                    return;
                }
                var match = matchChapterSource(target.title, chapters);
                if (!match) {
                    return;
                }
                var container = document.createElement('span');
                container.className = 'acrla-inline-remediation';
                container.setAttribute('data-acrla-inline-concept', match.concept);

                var button = document.createElement('button');
                button.type = 'button';
                button.className = 'acrla-inline-chapter-open';
                copyAcrlaDataAttributes(match.button, button);
                bindInlineChapterOpen(button);
                container.appendChild(button);
                target.appendTarget.appendChild(container);
                if (container.closest('a')) {
                    target.anchor.insertAdjacentElement('afterend', container);
                }
                updateMasteryButtonVisual(button, Number(button.getAttribute('data-acrla-score') || match.score || 0));
            });
        }

        function bindInlineChapterOpen(button) {
            button.addEventListener('click', function(event) {
                event.preventDefault();
                event.stopPropagation();
                if (event.stopImmediatePropagation) {
                    event.stopImmediatePropagation();
                }
                if (window.console && window.console.log) {
                    window.console.log('[ACRLA inline] chapter mastery clicked', button.dataset);
                }
                openAcrlaChatPanelForChapter(button);
                return false;
            }, true);
            ['pointerdown', 'mousedown', 'mouseup'].forEach(function(type) {
                button.addEventListener(type, function(event) {
                    event.preventDefault();
                    event.stopPropagation();
                    if (event.stopImmediatePropagation) {
                        event.stopImmediatePropagation();
                    }
                }, true);
            });
            button.addEventListener('keydown', function(event) {
                if (event.key !== 'Enter' && event.key !== ' ') {
                    return;
                }
                if (window.console && window.console.log) {
                    window.console.log('[ACRLA inline] chapter mastery clicked', button.dataset);
                }
                event.preventDefault();
                event.stopPropagation();
                if (event.stopImmediatePropagation) {
                    event.stopImmediatePropagation();
                }
                openAcrlaChatPanelForChapter(button);
                return false;
            }, true);
        }

        function chapterRemediationSources() {
            return Array.prototype.map.call(
                chapterGradeTemplates,
                function(button) {
                    var concept = button.getAttribute('data-acrla-concept') || button.getAttribute('data-acrla-label') || button.textContent || '';
                    return {
                        button: button,
                        concept: concept,
                        normalized: normalizeConcept(concept),
                        label: normalizeConcept(button.textContent || ''),
                        score: Number(button.getAttribute('data-acrla-score') || 0)
                    };
                }
            ).filter(function(item) {
                return item.normalized;
            });
        }

        function findMoodleChapterTargets() {
            var selectors = [
                '.activity a.aalink',
                '.activity .activityname a',
                '.activity .instancename',
                '.activity-item a.aalink',
                '.activity-item .activityname a',
                '.activity-item .instancename',
                'li.activity a[href]',
                '[data-activityname] a[href]',
                '[data-region="activity-card"] a[href]'
            ];
            var nodes = [];
            selectors.forEach(function(selector) {
                Array.prototype.forEach.call(document.querySelectorAll(selector), function(node) {
                    if (root.contains(node) || !isVisible(node)) {
                        return;
                    }
                    if (nodes.indexOf(node) === -1) {
                        nodes.push(node);
                    }
                });
            });
            return nodes.map(function(node) {
                var link = node.closest && node.closest('a') ? node.closest('a') : (node.tagName === 'A' ? node : null);
                var anchor = link || node;
                var row = anchor.closest('.activity, .activity-item, li, [data-activityname], [data-region="activity-card"]') || anchor.parentElement;
                var activity = anchor.closest('.activity, .activity-item, li.activity, .modtype_resource, li, [data-activityname], [data-region="activity-card"]') || row || anchor.parentElement;
                var appendTarget = activity.querySelector('.activityname, .activity-instance, .activitytitle, .media-body') || activity;
                if (appendTarget && appendTarget.closest && appendTarget.closest('a')) {
                    appendTarget = appendTarget.closest('a').parentElement || activity;
                }
                var title = (
                    node.getAttribute('title') ||
                    node.getAttribute('aria-label') ||
                    anchor.getAttribute('title') ||
                    anchor.getAttribute('aria-label') ||
                    node.textContent ||
                    row.getAttribute('data-activityname') ||
                    ''
                ).trim();
                return {anchor: anchor, row: row || anchor, appendTarget: appendTarget || row || anchor, title: title};
            }).filter(function(target) {
                return target.title && target.row && !root.contains(target.row);
            });
        }

        function matchChapterSource(title, sources) {
            var normalizedTitle = normalizeConcept(title);
            if (!normalizedTitle) {
                return null;
            }
            for (var i = 0; i < sources.length; i++) {
                if (conceptsMatch(normalizedTitle, sources[i].normalized) || conceptsMatch(normalizedTitle, sources[i].label)) {
                    return sources[i];
                }
            }
            return null;
        }

        function copyAcrlaDataAttributes(source, target) {
            Array.prototype.forEach.call(source.attributes, function(attr) {
                if (attr.name.indexOf('data-acrla-') === 0) {
                    target.setAttribute(attr.name, attr.value);
                }
            });
        }

        function updateInlineRemediationVisual(button, score) {
            var container = button.closest('.acrla-inline-remediation');
            if (!container) {
                return;
            }
            setMasteryClass(container, score);
            setMasteryClass(button, score);
            var scoreEl = button.querySelector('.acrla-inline-score');
            if (!scoreEl) {
                scoreEl = document.createElement('span');
                scoreEl.className = 'acrla-inline-score acrla-score-value';
                button.appendChild(scoreEl);
            }
            if (scoreEl) {
                scoreEl.textContent = formatScore(score) + '%';
                setMasteryClass(scoreEl, score);
            }
            var badge = button.querySelector('.acrla-inline-badge');
            if (!badge) {
                badge = document.createElement('span');
                badge.className = 'acrla-inline-badge';
                button.appendChild(badge);
            }
            if (badge) {
                setMasteryClass(badge, score);
                badge.innerHTML = '<span class="acrla-status-dot" aria-hidden="true"></span><span>' + getMasteryStatus(score) + '</span>';
            }
        }

        function startInlineRemediationObserver() {
            if (root.dataset.acrlaDashboard === '1' || !window.MutationObserver) {
                return;
            }
            var queued = false;
            var observer = new MutationObserver(function() {
                if (queued) {
                    return;
                }
                queued = true;
                window.setTimeout(function() {
                    queued = false;
                    renderInlineChapterRemediation();
                }, 300);
            });
            observer.observe(document.body, {childList: true, subtree: true});
        }

        function renderMasteryBadge(score) {
            return '<span class="acrla-mastery-badge ' + getMasteryClass(score) + '">' +
                '<span class="acrla-status-dot" aria-hidden="true"></span>' +
                '<span>' + getMasteryStatus(score) + '</span>' +
                '</span>';
        }

        function renderMasteryLegend() {
            if (root.dataset.acrlaDashboard !== '1' || document.getElementById('acrla-mastery-legend')) {
                return;
            }
            var legend = document.createElement('div');
            legend.id = 'acrla-mastery-legend';
            legend.className = 'acrla-mastery-legend';
            legend.innerHTML =
                '<h3>ACRLA Mastery Levels</h3>' +
                '<div class="acrla-legend-grid">' +
                '<span class="acrla-legend-item acrla-mastery-weak"><span class="acrla-status-dot"></span><strong>0 - 49%</strong><em>Weak</em></span>' +
                '<span class="acrla-legend-item acrla-mastery-moderate"><span class="acrla-status-dot"></span><strong>50 - 79%</strong><em>Moderate</em></span>' +
                '<span class="acrla-legend-item acrla-mastery-strong"><span class="acrla-status-dot"></span><strong>80 - 100%</strong><em>Strong</em></span>' +
                '</div>';
            var target = document.querySelector('.course-card, .dashboard-card, .card');
            var container = target && target.parentNode ? target.parentNode : (document.querySelector('main') || document.body);
            container.appendChild(legend);
        }

        function renderPanelMasteryOverview(data) {
            var overview = document.getElementById('acrla-mastery-overview');
            if (!overview || !data || data.status !== 'ok') {
                return;
            }
            var courses = data.courses || [];
            var selectedCourse = null;
            for (var i = 0; i < courses.length; i++) {
                if (String(courses[i].course_id || '') === String(activeCourseId || '')) {
                    selectedCourse = courses[i];
                    break;
                }
            }
            if (!selectedCourse && courses.length) {
                selectedCourse = courses[0];
                activeCourseId = String(selectedCourse.course_id || activeCourseId || '');
            }
            var overall = Number(data.overall_current_acrla_mastery || 0);
            var html =
                '<section class="acrla-overview-section">' +
                '<h3>Your Mastery Overview</h3>' +
                '<div class="acrla-overview-summary" data-acrla-mastery-score="' + escapeHtml(String(overall)) + '">' +
                '<div class="acrla-ring" style="--acrla-angle:' + (Math.max(0, Math.min(100, overall)) * 3.6) + 'deg"></div>' +
                '<div><span>Overall ACRLA Mastery</span><strong class="acrla-score-value">' + formatScore(overall) + '%</strong>' + renderMasteryBadge(overall) + '</div>' +
                '</div>' +
                '</section>';

            if (courses.length) {
                html += '<section class="acrla-overview-section"><h3>Course Mastery</h3><div class="acrla-overview-list">';
                courses.forEach(function(course) {
                    var score = Number(course.course_current_acrla_mastery || 0);
                    html += '<button type="button" class="acrla-overview-row" data-acrla-mastery-score="' + escapeHtml(String(score)) + '" data-acrla-overview-course="' + escapeHtml(String(course.course_id || '')) + '">' +
                        '<span class="acrla-row-icon" aria-hidden="true">' + courseIcon(course.course_name || '') + '</span>' +
                        '<span class="acrla-row-title">' + escapeHtml(course.course_name || 'Course') + '</span>' +
                        '<strong class="acrla-score-value">' + formatScore(score) + '%</strong>' +
                        renderMasteryBadge(score) +
                        '</button>';
                });
                html += '</div></section>';
            }

            if (selectedCourse) {
                html += '<section class="acrla-overview-section"><h3>Chapter Mastery (' + escapeHtml(selectedCourse.course_name || 'Selected course') + ')</h3><div class="acrla-overview-list">';
                var concepts = selectedCourse.concept_current_acrla_mastery || {};
                Object.keys(concepts).forEach(function(concept) {
                    var score = Number(concepts[concept].current_acrla_mastery || 0);
                    html += '<button type="button" class="acrla-overview-row acrla-overview-chapter" data-acrla-mastery-score="' + escapeHtml(String(score)) + '" data-acrla-overview-concept="' + escapeHtml(concept) + '">' +
                        '<span class="acrla-row-icon acrla-row-icon-chapter" aria-hidden="true">&#128214;</span>' +
                        '<span class="acrla-row-title">' + escapeHtml(concept) + '</span>' +
                        '<strong class="acrla-score-value">' + formatScore(score) + '%</strong>' +
                        renderMasteryBadge(score) +
                        '</button>';
                });
                html += '</div></section>';
            }

            html += '<div class="acrla-overview-legend">' +
                '<strong>Legend</strong>' +
                '<span class="acrla-mastery-weak"><span class="acrla-status-dot"></span>0-49 Weak</span>' +
                '<span class="acrla-mastery-moderate"><span class="acrla-status-dot"></span>50-79 Moderate</span>' +
                '<span class="acrla-mastery-strong"><span class="acrla-status-dot"></span>80-100 Strong</span>' +
                '</div>';
            overview.innerHTML = html;
            Array.prototype.forEach.call(overview.querySelectorAll('[data-acrla-mastery-score]'), function(element) {
                setMasteryClass(element, Number(element.getAttribute('data-acrla-mastery-score') || 0));
            });
            Array.prototype.forEach.call(overview.querySelectorAll('[data-acrla-overview-course]'), function(row) {
                row.addEventListener('click', function() {
                    activeCourseId = row.getAttribute('data-acrla-overview-course') || activeCourseId;
                    renderPanelMasteryOverview(latestMasteryData);
                });
            });
        }

        function formatScore(score) {
            var rounded = Math.round(Number(score || 0) * 100) / 100;
            return rounded.toFixed(2).replace(/\.?0+$/, '');
        }

        function courseIcon(name) {
            var n = normalizeText(name);
            if (n.indexOf('math') !== -1) {
                return '&pi;';
            }
            if (n.indexOf('data') !== -1) {
                return '&#9635;';
            }
            if (n.indexOf('computer') !== -1 || n.indexOf('science') !== -1) {
                return '&#9635;';
            }
            return '&#9679;';
        }

        function escapeHtml(value) {
            return String(value || '')
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        }

        function findCardByVisibleTitle(courseName) {
            var normalizedName = normalizeText(courseName);
            var cards = Array.prototype.slice.call(document.querySelectorAll('.card'));
            for (var i = 0; i < cards.length; i++) {
                var card = cards[i];
                if (root.contains(card)) {
                    continue;
                }
                if (normalizeText(card.innerText || card.textContent).indexOf(normalizedName) !== -1) {
                    return card;
                }
            }
            return null;
        }

        function courseEntriesFromDataAttributes() {
            var entries = [];
            var seen = {};
            Array.prototype.forEach.call(document.querySelectorAll('[data-course-id]'), function(element) {
                if (root.contains(element)) {
                    return;
                }
                var courseId = String(element.dataset.courseId || element.getAttribute('data-course-id') || '');
                if (!courseId || seen[courseId]) {
                    return;
                }
                var card = findCourseCardFromElement(element);
                if (!card) {
                    return;
                }
                entries.push({
                    card: card,
                    link: null,
                    courseId: courseId,
                    courseName: (element.innerText || element.title || '').trim()
                });
                seen[courseId] = true;
            });
            return entries;
        }

        function dashboardCourseButtonMap() {
            var buttons = {};
            Array.prototype.forEach.call(root.querySelectorAll('.acrla-dashboard-course-button'), function(button) {
                var courseId = String(button.getAttribute('data-acrla-course-id') || '');
                if (courseId) {
                    buttons[courseId] = button;
                }
            });
            return buttons;
        }

        function logDomDiagnostics() {
            if (!window.console || !window.console.log) {
                return;
            }
            window.console.log('[ACRLA] page', document.body ? document.body.id : '');
            window.console.log('[ACRLA] all cards', document.querySelectorAll('.card').length);
            window.console.log('[ACRLA] all course card candidates', document.querySelectorAll('[data-region]').length);
            window.console.log('[ACRLA] data-course-id elements', document.querySelectorAll('[data-course-id]').length);
            Array.prototype.forEach.call(document.querySelectorAll('.card'), function(card, index) {
                if (index < 10) {
                    window.console.log('[ACRLA] card text', index, card.innerText);
                }
            });
        }

        function findCourseOverviewTarget() {
            var headings = Array.prototype.slice.call(document.querySelectorAll('h1, h2, h3, h4'));
            for (var i = 0; i < headings.length; i++) {
                if (/course overview|my courses/i.test(headings[i].textContent || '')) {
                    return headings[i].nextElementSibling || headings[i];
                }
            }
            var firstCourseLink = document.querySelector('a[href*="/course/view.php?id="], a[href*="course/view.php?id="]');
            var firstCard = firstCourseLink ? findCourseCard(firstCourseLink) : null;
            return firstCard || document.querySelector('main') || document.body.firstElementChild || document.body;
        }

        function extractCourseId(href) {
            if (!href) {
                return '';
            }
            try {
                var url = new URL(href, window.location.origin);
                return url.searchParams.get('id') || '';
            } catch (error) {
                var match = String(href).match(/[?&]id=(\d+)/);
                return match ? match[1] : '';
            }
        }

        function findCourseCard(courseLink) {
            var closest = courseLink.closest(
                '[data-region="course-content"], ' +
                '[data-region="course-card"], ' +
                '[data-course-id], ' +
                '.course-card, ' +
                '.dashboard-card, ' +
                '.coursebox, ' +
                '.card, ' +
                'li'
            );
            if (closest && isVisible(closest)) {
                return closest;
            }

            var node = courseLink.parentElement;
            var steps = 0;
            while (node && node !== document.body && steps < 8) {
                if (isVisible(node) && looksLikeCourseContainer(node)) {
                    return node;
                }
                node = node.parentElement;
                steps += 1;
            }
            return courseLink.parentElement;
        }

        function findCourseCardFromElement(element) {
            var node = element;
            var steps = 0;
            while (node && node !== document.body && steps < 10) {
                if (root.contains(node)) {
                    return null;
                }
                if (isVisible(node) && looksLikeCourseContainer(node)) {
                    return node;
                }
                if (isVisible(node) && node.querySelector && node.querySelector('[data-course-id]')) {
                    return node;
                }
                node = node.parentElement;
                steps += 1;
            }
            return element.parentElement;
        }

        function findCourseCardByName(courseName) {
            var name = normalizeText(courseName);
            if (!name) {
                return null;
            }
            var candidates = Array.prototype.slice.call(document.querySelectorAll(
                '[data-course-id], ' +
                '[data-courseid], ' +
                '[data-region="course-content"], ' +
                '[data-region="course-card"], ' +
                '.course-card, ' +
                '.dashboard-card, ' +
                '.coursebox, ' +
                '.card, ' +
                'li'
            ));
            for (var i = 0; i < candidates.length; i++) {
                var candidate = candidates[i];
                if (root.contains(candidate) || !isVisible(candidate)) {
                    continue;
                }
                if (normalizeText(candidate.textContent).indexOf(name) !== -1) {
                    return candidate;
                }
            }
            return null;
        }

        function insertCourseCardButton(card, button) {
            var target = card.querySelector(
                '.coursename, ' +
                '.course-name, ' +
                '.multiline, ' +
                '.course-info-container, ' +
                '.dashboard-card-deck, ' +
                '.dashboard-card-footer, ' +
                '.card-footer, ' +
                '.card-body, ' +
                '[data-region="course-info"], ' +
                '[data-region="course-content"]'
            ) || card;
            target.appendChild(button);
        }

        function looksLikeCourseContainer(node) {
            if (!node || !node.querySelector) {
                return false;
            }
            var text = (node.textContent || '').trim();
            return text.length > 0 && Boolean(node.querySelector('a[href*="course/view.php?id="]'));
        }

        function normalizeText(value) {
            return String(value || '').replace(/\s+/g, ' ').trim().toLowerCase();
        }

        function addQueryParams(url, params) {
            if (!url) {
                return '';
            }
            try {
                var parsed = new URL(url, window.location.href);
                Object.keys(params).forEach(function(key) {
                    var value = params[key];
                    if (value !== undefined && value !== null && String(value) !== '') {
                        parsed.searchParams.set(key, value);
                    }
                });
                return parsed.toString();
            } catch (error) {
                var query = new URLSearchParams();
                Object.keys(params).forEach(function(key) {
                    var value = params[key];
                    if (value !== undefined && value !== null && String(value) !== '') {
                        query.set(key, value);
                    }
                });
                return url + (url.indexOf('?') === -1 ? '?' : '&') + query.toString();
            }
        }

        function cleanChapterTitle(value) {
            var cleaned = String(value || 'this chapter')
                .replace(/\.pdf$/i, '')
                .replace(/[\-_]/g, ' ')
                .replace(/\b(chapter|chap|ch)\s*\d+\b[:\.\-\s]*/gi, '')
                .replace(/\s+/g, ' ')
                .trim();
            var normalized = normalizeConcept(cleaned);
            var labels = {
                'recursion': 'Recursion',
                'sorting': 'Sorting Algorithms',
                'sorting algorithms': 'Sorting Algorithms',
                'pointers memory': 'Pointers and Memory Management',
                'pointers and memory': 'Pointers and Memory Management',
                'pointers and memory management': 'Pointers and Memory Management',
                'binary trees': 'Binary Trees and BSTs',
                'binary trees and bsts': 'Binary Trees and BSTs',
                'bst': 'Binary Trees and BSTs',
                'bsts': 'Binary Trees and BSTs'
            };
            if (labels[normalized]) {
                return labels[normalized];
            }
            return cleaned.replace(/\b\w/g, function(letter) {
                return letter.toUpperCase();
            }) || 'This Chapter';
        }

        // Mirrors PHP local_acrla_clean_concept_title: removes .pdf suffix,
        // replaces _/- with spaces, strips "chapter N" prefixes, lowercases.
        // Used to match PHP-rendered data-acrla-concept values against API keys.
        function normalizeConcept(s) {
            return String(s)
                .replace(/\.pdf$/i, '')
                .replace(/[\-_]/g, ' ')
                .replace(/\b(chapter|chap|ch)\s*\d+\b[:\.\-\s]*/gi, '')
                .toLowerCase()
                .replace(/\s+/g, ' ')
                .trim();
        }

        // Three-tier alias-aware concept matching.
        // Tier 1: exact normalized string ("sets" === "sets").
        // Tier 2: substring containment ("binary trees" ⊂ "binary trees and bsts").
        // Tier 3: word-set subset — every word of the shorter string appears in
        //         the longer ("pointers memory" ⊆ "pointers and memory management").
        // This avoids hard-coding aliases while handling the real mismatches:
        //   sorting           ↔ sorting algorithms
        //   pointers memory   ↔ pointers and memory management
        //   binary trees      ↔ binary trees and bsts
        function conceptsMatch(a, b) {
            if (!a || !b) { return false; }
            if (a === b) { return true; }
            if (a.indexOf(b) !== -1 || b.indexOf(a) !== -1) { return true; }
            var wa = a.split(' ').filter(function(w) { return w.length > 1; });
            var wb = b.split(' ').filter(function(w) { return w.length > 1; });
            if (wa.length === 0 || wb.length === 0) { return false; }
            var shorter = wa.length <= wb.length ? wa : wb;
            var longer  = wa.length <= wb.length ? wb : wa;
            return shorter.every(function(w) { return longer.indexOf(w) !== -1; });
        }

        // Apply a mastery map (raw key → percent) from the sync response to all
        // chapter buttons for the given course, using normalized concept matching.
        function applySyncMastery(courseId, masteryMap) {
            var syncList = [];
            Object.keys(masteryMap).forEach(function(key) {
                var v = Number(masteryMap[key] || 0);
                if (v > 0) {
                    syncList.push({norm: normalizeConcept(key), val: v, orig: key});
                }
            });
            Array.prototype.forEach.call(
                document.querySelectorAll('.acrla-grade-link[data-acrla-level-type="chapter"][data-acrla-course-id="' + cssEscape(courseId) + '"]'),
                function(button) {
                    var btnConcept = button.getAttribute('data-acrla-concept') || '';
                    var btnNorm = normalizeConcept(btnConcept);
                    var match = null;
                    for (var i = 0; i < syncList.length; i++) {
                        if (conceptsMatch(btnNorm, syncList[i].norm)) {
                            match = syncList[i];
                            break;
                        }
                    }
                    if (match !== null) {
                        if (window.console && window.console.log) {
                            window.console.log(
                                '[ACRLA] chapter_mastery_display ' +
                                'button_concept=' + btnConcept + ' ' +
                                'normalized_concept_key=' + btnNorm + ' ' +
                                'matched_backend_key=' + match.orig + ' ' +
                                'concept_current_acrla_mastery=' + match.val + ' ' +
                                'course_current_acrla_mastery=not_used_sync_response ' +
                                'value_written_strong=' + (Math.round(match.val * 10) / 10) + '% ' +
                                'value_written_data_acrla_score=' + (Math.round(match.val * 10) / 10)
                            );
                        }
                        updateGradeButton(button, match.val);
                    }
                }
            );
        }

        function isVisible(node) {
            if (!node || !node.getBoundingClientRect) {
                return false;
            }
            var rect = node.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
        }

        function cssEscape(value) {
            if (window.CSS && typeof window.CSS.escape === 'function') {
                return window.CSS.escape(String(value));
            }
            return String(value).replace(/"/g, '\\"');
        }

        function setStatus(text) {
            if (status) {
                status.textContent = text;
            }
        }

        function runFinally(promise, callback) {
            if (promise && typeof promise.finally === 'function') {
                promise.finally(callback);
                return;
            }
            Promise.resolve(promise).then(callback, callback);
        }
    }

    function parsePayload(raw) {
        try {
            return JSON.parse(raw);
        } catch (error) {
            return {};
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAcrlaWidget);
    } else {
        initAcrlaWidget();
    }
})();
