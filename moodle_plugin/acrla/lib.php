<?php
// This file is part of ACRLA.
//
// ACRLA Moodle local plugin integration.
//
// Purpose:
//   Injects the ACRLA launcher, side panel, mastery controls, and sync payloads
//   into Moodle dashboard/course pages.
//
// Main responsibilities:
//   - provide Moodle user/course context to the JavaScript widget
//   - build student/course/mastery payloads for the FastAPI backend
//   - render grade/remediation trigger markup with stable data attributes
//   - avoid direct database access from ACRLA into Moodle

defined('MOODLE_INTERNAL') || die();

// ==========================================================
// Widget Injection
// ==========================================================

/**
 * Inject the ACRLA launcher into Moodle course pages.
 *
 * The Moodle plugin does not connect to the ACRLA database. It sends the
 * current Moodle context to the ACRLA backend through /api/v1/moodle/sync.
 */
function local_acrla_before_footer() {
    global $USER, $COURSE, $PAGE, $CFG;

    $config = get_config('local_acrla');
    if (empty($config->enabled)) {
        return '';
    }

    if (!isloggedin() || isguestuser()) {
        return '';
    }

    if (strpos($PAGE->pagetype, 'login') !== false) {
        return '';
    }

    $isdashboard = local_acrla_is_dashboard_page($PAGE);
    $iscoursepage = !empty($COURSE->id) && $COURSE->id != SITEID;
    if (!$isdashboard && !$iscoursepage) {
        return '';
    }

    $backendurl = !empty($config->backend_url)
        ? rtrim($config->backend_url, '/')
        : 'http://localhost:8000';

    $contextcourse = $iscoursepage ? $COURSE : local_acrla_get_default_dashboard_course();
    // The PHP layer prepares Moodle context as data attributes. JavaScript then
    // sends the payload to FastAPI; ACRLA never reads Moodle's database itself.
    $payload = local_acrla_build_sync_payload($USER, $contextcourse);
    $dashboardpayloads = $isdashboard ? local_acrla_build_dashboard_sync_payloads($USER) : [];
    $iframeurl = local_acrla_build_iframe_url($backendurl, $USER, $contextcourse);
    $launchurl = $iscoursepage ? local_acrla_build_remediation_launch_url($backendurl, $USER, $COURSE) : '';
    [$launchconcept] = $iscoursepage ? local_acrla_get_weakest_concept($USER, $COURSE) : ['', 0];
    $gradelinks = $iscoursepage
        ? local_acrla_build_chapter_remediation_links($USER, $COURSE)
        : local_acrla_build_dashboard_remediation_links($USER);
    $materialsyncurl = new moodle_url('/local/acrla/sync_materials.php', [
        'courseid' => (int)$contextcourse->id,
        'sesskey' => sesskey(),
    ]);

    $assetversion = '2026062601';
    $cssurl = new moodle_url('/local/acrla/styles.css', ['v' => $assetversion]);
    $jsurl = new moodle_url('/local/acrla/widget.js', ['v' => $assetversion]);

    $payloadjson = json_encode($payload, JSON_HEX_TAG | JSON_HEX_APOS | JSON_HEX_AMP | JSON_HEX_QUOT);
    $dashboardpayloadsjson = json_encode($dashboardpayloads, JSON_HEX_TAG | JSON_HEX_APOS | JSON_HEX_AMP | JSON_HEX_QUOT);

    return '
<link rel="stylesheet" href="' . s($cssurl->out(false)) . '">
<div id="acrla-widget"
     class="' . ($isdashboard ? 'acrla-dashboard-widget' : 'acrla-course-widget') . '"
     data-backend-url="' . s($backendurl) . '"
     data-iframe-url="' . s($iframeurl) . '"
     data-sync-payload="' . s($payloadjson) . '"
     data-dashboard-sync-payloads="' . s($dashboardpayloadsjson) . '"
     data-material-sync-url="' . s($materialsyncurl->out(false)) . '"
     data-acrla-dashboard="' . ($isdashboard ? '1' : '0') . '">
    <button type="button" id="acrla-launcher" aria-label="' . s(get_string('openchat', 'local_acrla')) . '">
        ACRLA
    </button>
    <button type="button" id="acrla-material-sync">
        ' . s(get_string('syncmaterials', 'local_acrla')) . '
    </button>
    ' . ($iscoursepage ? '<a id="acrla-remediation-link" href="' . s($launchurl) . '" data-launch-url="' . s($launchurl) . '">
        Open ACRLA remediation for ' . s($launchconcept) . '
    </a>' : '') . '
    <div id="acrla-grade-remediation">
        <strong>' . ($isdashboard ? 'ACRLA mastery by course' : 'ACRLA chapter remediation') . '</strong>
        ' . $gradelinks . '
    </div>
    <div id="acrla-panel" aria-hidden="true">
        <div id="acrla-panel-header">
            <div>
                <strong>ACRLA</strong>
                <span>Adaptive Conversation &amp; Remediation</span>
                <span id="acrla-sync-status">' . s(get_string('ready', 'local_acrla')) . '</span>
            </div>
            <button type="button" id="acrla-close" aria-label="' . s(get_string('closechat', 'local_acrla')) . '">x</button>
        </div>
        <div id="acrla-mastery-overview"></div>
        <iframe id="acrla-frame" title="ACRLA Learning Assistant" loading="lazy"></iframe>
    </div>
</div>
<script src="' . s($jsurl->out(false)) . '"></script>';
}

// ==========================================================
// Moodle Context and Sync Payloads
// ==========================================================

/**
 * Build the payload sent to ACRLA's Moodle sync endpoint.
 *
 * Build a dynamic Moodle-to-ACRLA profile payload.
 */
function local_acrla_build_sync_payload(stdClass $user, stdClass $course): array {
    // This is the profile/mastery snapshot sent to `/api/v1/moodle/sync`.
    // For the MVP, chapter mastery can come from demo mappings or Moodle
    // resource-derived values; the backend stores it as the initial baseline.
    return [
        'student_id' => (int)$user->id,
        'course_id' => (int)$course->id,
        'student_name' => fullname($user),
        'course_name' => format_string($course->fullname),
        'difficulty' => 'easy',
        'mastery' => local_acrla_get_course_mastery($user, $course),
    ];
}

/**
 * Build one sync payload per dashboard course so overall/course launches have
 * the latest per-course mastery in ACRLA before opening remediation.
 */
function local_acrla_build_dashboard_sync_payloads(stdClass $user): array {
    $payloads = [];
    foreach (local_acrla_get_student_courses() as $course) {
        $payloads[] = local_acrla_build_sync_payload($user, $course);
    }
    return $payloads;
}

/**
 * Detect Moodle dashboard / My courses pages.
 */
function local_acrla_is_dashboard_page(moodle_page $page): bool {
    $pagetype = (string)$page->pagetype;
    return $pagetype === 'my-index'
        || $pagetype === 'my-courses'
        || strpos($pagetype, 'my-') === 0;
}

/**
 * Get courses visible to the current student for dashboard-level buttons.
 */
function local_acrla_get_student_courses(): array {
    if (function_exists('enrol_get_my_courses')) {
        $courses = enrol_get_my_courses(['id', 'fullname', 'shortname']);
        return array_values($courses);
    }
    return [];
}

/**
 * Fallback context for dashboard-only actions.
 */
function local_acrla_get_default_dashboard_course(): stdClass {
    $courses = local_acrla_get_student_courses();
    if ($courses) {
        return $courses[0];
    }
    $course = new stdClass();
    $course->id = SITEID;
    $course->fullname = 'All courses';
    $course->shortname = 'All courses';
    return $course;
}

/**
 * Discover ACRLA concepts for a Moodle course from synced/course material.
 *
 * The MVP treats each PDF resource title as a chapter/concept. Teachers can add
 * a new course by uploading PDFs with meaningful resource names; no plugin code
 * change is needed.
 */
function local_acrla_get_course_concepts(stdClass $course): array {
    global $DB;

    $concepts = [];
    $modules = $DB->get_records_sql(
        "SELECT cm.id AS cmid, r.name
           FROM {course_modules} cm
           JOIN {modules} m ON m.id = cm.module
           JOIN {resource} r ON r.id = cm.instance
          WHERE cm.course = ?
            AND m.name = ?",
        [(int)$course->id, 'resource']
    );

    foreach ($modules as $module) {
        $title = format_string($module->name);
        $concept = local_acrla_clean_concept_title($title);
        if ($concept !== '' && !isset($concepts[$concept])) {
            $concepts[$concept] = $concept;
        }
    }

    return array_values($concepts);
}

/**
 * Build mastery for every discovered course concept.
 */
function local_acrla_get_course_mastery(stdClass $user, stdClass $course): array {
    $mastery = [];
    foreach (local_acrla_get_course_concepts($course) as $concept) {
        $demo = local_acrla_demo_score_for_concept($course, $concept);
        $mastery[$concept] = $demo !== null
            ? $demo
            : local_acrla_score_for_concept($user, $course, $concept, 50.0);
    }

    if (!$mastery) {
        $fallback = local_acrla_clean_concept_title(format_string($course->fullname));
        if ($fallback !== '') {
            $demo = local_acrla_demo_score_for_concept($course, $fallback);
            $mastery[$fallback] = $demo !== null
                ? $demo
                : local_acrla_score_for_concept($user, $course, $fallback, 50.0);
        }
    }

    return $mastery;
}

/**
 * Fixed thesis-demo Moodle mastery baselines.
 */
function local_acrla_demo_score_for_concept(stdClass $course, string $concept): ?float {
    $course_name = local_acrla_normalize_label(format_string($course->fullname));
    $concept_name = local_acrla_normalize_label($concept);

    if (strpos($course_name, 'data science') !== false) {
        return 55.0;
    }

    if (strpos($course_name, 'computer science') !== false) {
        $scores = [
            'recursion' => 40.0,
            'sorting algorithms' => 50.0,
            'sorting' => 50.0,
            'pointers and memory management' => 55.0,
            'pointers memory' => 55.0,
            'pointers and memory' => 55.0,
            'binary trees and bsts' => 60.0,
            'binary trees' => 60.0,
            'binary tree' => 60.0,
        ];
        return $scores[$concept_name] ?? null;
    }

    if (strpos($course_name, 'mathematics') !== false || $course_name === 'math') {
        $scores = [
            'logic' => 76.0,
            'sets' => 88.0,
            'graphs' => 90.0,
            'relations functions' => 33.0,
        ];
        return $scores[$concept_name] ?? null;
    }

    return null;
}

function local_acrla_normalize_label(string $value): string {
    $value = strtolower($value);
    $value = preg_replace('/[^a-z0-9]+/', ' ', $value);
    return trim(preg_replace('/\s+/', ' ', $value));
}

/**
 * Resolve a concept score from Moodle gradebook where possible.
 */
function local_acrla_score_for_concept(stdClass $user, stdClass $course, string $concept, float $default): float {
    global $DB;

    $like = '%' . $DB->sql_like_escape($concept) . '%';
    $records = $DB->get_records_sql(
        "SELECT gi.id, gi.grademax, gg.finalgrade
           FROM {grade_items} gi
      LEFT JOIN {grade_grades} gg ON gg.itemid = gi.id AND gg.userid = ?
          WHERE gi.courseid = ?
            AND gi.itemname IS NOT NULL
            AND " . $DB->sql_like('gi.itemname', '?', false) . "
          ORDER BY gi.id DESC",
        [(int)$user->id, (int)$course->id, $like],
        0,
        1
    );

    if ($records) {
        $record = reset($records);
        if ($record && $record->finalgrade !== null && (float)$record->grademax > 0) {
            return max(0.0, min(100.0, ((float)$record->finalgrade / (float)$record->grademax) * 100.0));
        }
    }

    return $default;
}

/**
 * Clean a Moodle resource/course title into a displayable concept name.
 */
function local_acrla_clean_concept_title(string $title): string {
    $title = preg_replace('/\.[Pp][Dd][Ff]$/', '', $title);
    $title = str_replace(['_', '-'], ' ', $title);
    $title = preg_replace('/\b(chapter|chap|ch)\s*\d+\b[:.\-\s]*/i', '', $title);
    $title = preg_replace('/\s+/', ' ', trim($title));
    return $title;
}

/**
 * Pick the lowest-scoring concept for the course remediation link.
 */
function local_acrla_get_weakest_concept(stdClass $user, stdClass $course): array {
    $mastery = local_acrla_get_course_mastery($user, $course);
    asort($mastery, SORT_NUMERIC);

    $concept = (string)array_key_first($mastery);
    return [$concept, (float)$mastery[$concept]];
}

/**
 * Course mastery is the simple average of that course's chapter/concept scores.
 */
function local_acrla_course_score(stdClass $user, stdClass $course): float {
    return local_acrla_average_score(local_acrla_get_course_mastery($user, $course));
}

/**
 * Build the iframe URL for the standalone ACRLA frontend.
 */
function local_acrla_build_iframe_url(string $backendurl, stdClass $user, stdClass $course): string {
    $params = [
        'student_id' => (int)$user->id,
        'course_id' => (int)$course->id,
        'course_name' => format_string($course->fullname),
        'source' => 'moodle',
    ];
    return $backendurl . '/?' . http_build_query($params, '', '&', PHP_QUERY_RFC3986);
}

/**
 * Build a clickable remediation launch URL for the weakest discovered concept.
 */
function local_acrla_build_remediation_launch_url(string $backendurl, stdClass $user, stdClass $course): string {
    // Course pages keep this URL as a fallback/remediation context. The modern
    // UI usually opens the same target through the persistent side panel.
    [$concept, $score] = local_acrla_get_weakest_concept($user, $course);

    $params = [
        'student_id' => (int)$user->id,
        'course_id' => (int)$course->id,
        'student_name' => fullname($user),
        'course_name' => format_string($course->fullname),
        'concept' => $concept,
        'score' => $score,
        'level_type' => 'chapter',
        'source' => 'moodle',
    ];
    return $backendurl . '/api/v1/moodle/launch?' . http_build_query($params, '', '&', PHP_QUERY_RFC3986);
}

/**
 * Build clickable dashboard-level remediation buttons.
 *
 * Course buttons are built from the student's discovered Moodle courses.
 */
function local_acrla_build_dashboard_remediation_links(stdClass $user): string {
    $links = [];
    $courses = local_acrla_get_student_courses();
    if (!$courses) {
        $courses = [local_acrla_get_default_dashboard_course()];
    }

    $course_scores = [];
    foreach ($courses as $course) {
        $score = local_acrla_course_score($user, $course);
        $course_scores[] = $score;
    }
    $overall_score = local_acrla_average_score($course_scores);

    $links[] = local_acrla_grade_button(
        $user,
        local_acrla_get_default_dashboard_course(),
        'Overall ACRLA mastery',
        $overall_score,
        'overall'
    );

    foreach ($courses as $course) {
        $links[] = local_acrla_grade_button(
            $user,
            $course,
            'ACRLA course mastery',
            local_acrla_course_score($user, $course),
            'course',
            '',
            'acrla-dashboard-course-button'
        );
    }

    return implode('', $links);
}

/**
 * Build clickable chapter grades for an individual Moodle course page only.
 */
function local_acrla_build_chapter_remediation_links(stdClass $user, stdClass $course): string {
    $links = [];
    $mastery = local_acrla_get_course_mastery($user, $course);
    foreach ($mastery as $concept => $score) {
        $links[] = local_acrla_grade_button($user, $course, $concept, $score, 'chapter', $concept);
    }
    return implode('', $links);
}

/**
 * Render one clickable ACRLA grade remediation button.
 */
function local_acrla_grade_button(
    stdClass $user,
    stdClass $course,
    string $label,
    float $score,
    string $level_type,
    string $concept = '',
    string $extra_class = ''
): string {
    // These data attributes are the contract between Moodle markup and
    // widget.js. They allow one click handler to launch overall, course, or
    // chapter remediation without hardcoding course names in JavaScript.
    $classes = trim('acrla-grade-link ' . $extra_class);
    return '
        <button type="button"
                class="' . s($classes) . '"
                data-acrla-student-id="' . s((string)(int)$user->id) . '"
                data-acrla-course-id="' . s((string)(int)$course->id) . '"
                data-acrla-student-name="' . s(fullname($user)) . '"
                data-acrla-course-name="' . s(format_string($course->fullname)) . '"
                data-acrla-concept="' . s($concept) . '"
                data-acrla-score="' . s((string)$score) . '"
                data-acrla-level-type="' . s($level_type) . '">
            <span>' . s($label) . '</span>
            <strong>' . s((string)round($score, 1)) . '%</strong>
        </button>';
}

/**
 * Compute a demo average score.
 */
function local_acrla_average_score(array $scores): float {
    if (!$scores) {
        return 0.0;
    }
    return array_sum($scores) / count($scores);
}
