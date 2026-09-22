<?php
// This file is part of ACRLA.

define('AJAX_SCRIPT', true);

require_once(__DIR__ . '/../../config.php');
require_once($CFG->libdir . '/filelib.php');

$courseid = required_param('courseid', PARAM_INT);

require_sesskey();
$course = get_course($courseid);
require_login($course);

$config = get_config('local_acrla');
$backendurl = !empty($config->backend_url)
    ? rtrim($config->backend_url, '/')
    : 'http://localhost:8000';

try {
    $results = local_acrla_sync_course_pdfs($backendurl, $course);
    local_acrla_materials_json([
        'status' => 'ok',
        'course_id' => (int)$course->id,
        'files_synced' => count($results),
        'results' => $results,
    ]);
} catch (Throwable $error) {
    local_acrla_materials_json([
        'status' => 'error',
        'message' => $error->getMessage(),
    ], 500);
}

/**
 * Locate PDF resource files in the current Moodle course and send them to ACRLA.
 *
 * TODO: Expand this to folders, pages, books, and concept-aware gradebook mappings.
 *
 * @param string $backendurl ACRLA backend base URL.
 * @param stdClass $course Moodle course record.
 * @return array Sync results returned by the ACRLA backend.
 */
function local_acrla_sync_course_pdfs(string $backendurl, stdClass $course): array {
    global $DB, $USER, $CFG;

    $fs = get_file_storage();
    $modules = $DB->get_records_sql(
        "SELECT cm.id AS cmid, r.name
           FROM {course_modules} cm
           JOIN {modules} m ON m.id = cm.module
           JOIN {resource} r ON r.id = cm.instance
          WHERE cm.course = ?
            AND m.name = ?",
        [(int)$course->id, 'resource']
    );

    $tmpdir = make_temp_directory('local_acrla');
    $results = [];

    foreach ($modules as $module) {
        $context = context_module::instance($module->cmid);
        $files = $fs->get_area_files($context->id, 'mod_resource', 'content', 0, 'sortorder', false);

        foreach ($files as $file) {
            if ($file->is_directory()) {
                continue;
            }

            $filename = $file->get_filename();
            $extension = strtolower(pathinfo($filename, PATHINFO_EXTENSION));
            $mimetype = $file->get_mimetype();
            if ($extension !== 'pdf' && $mimetype !== 'application/pdf') {
                continue;
            }

            $tmpfile = $tmpdir . '/' . clean_param($module->cmid . '-' . $filename, PARAM_FILE);
            $file->copy_content_to($tmpfile);

            try {
                $results[] = local_acrla_upload_pdf_to_backend(
                    $backendurl,
                    $course,
                    (int)$USER->id,
                    $tmpfile,
                    $filename,
                    format_string($module->name),
                    local_acrla_infer_concept($module->name ?: $filename)
                );
            } finally {
                @unlink($tmpfile);
            }
        }
    }

    return $results;
}

/**
 * Upload one PDF to the ACRLA backend material ingestion endpoint.
 *
 * @param string $backendurl ACRLA backend base URL.
 * @param stdClass $course Moodle course record.
 * @param int $studentid Current Moodle user id.
 * @param string $filepath Local temp PDF path.
 * @param string $filename Original Moodle filename.
 * @param string $resourcetitle Moodle resource title shown to students.
 * @return array Decoded backend response with filename attached.
 */
function local_acrla_upload_pdf_to_backend(
    string $backendurl,
    stdClass $course,
    int $studentid,
    string $filepath,
    string $filename,
    string $resourcetitle,
    ?string $concept = null
): array {
    if (!function_exists('curl_init') || !function_exists('curl_file_create')) {
        throw new RuntimeException('PHP cURL file uploads are not available.');
    }

    $endpoint = $backendurl . '/api/v1/moodle/materials/sync';
    $handle = curl_init($endpoint);
    $postfields = [
        'student_id' => (string)$studentid,
        'course_id' => (string)$course->id,
        'course_name' => format_string($course->fullname),
        'original_file_name' => $filename,
        'moodle_resource_title' => $resourcetitle,
        'file' => curl_file_create($filepath, 'application/pdf', $filename),
    ];
    if ($concept) {
        $postfields['concept'] = $concept;
    }

    curl_setopt($handle, CURLOPT_POST, true);
    curl_setopt($handle, CURLOPT_POSTFIELDS, $postfields);
    curl_setopt($handle, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($handle, CURLOPT_TIMEOUT, 120);

    $body = curl_exec($handle);
    $error = curl_error($handle);
    $status = curl_getinfo($handle, CURLINFO_HTTP_CODE);
    curl_close($handle);

    if ($body === false || $error) {
        throw new RuntimeException('ACRLA material upload failed: ' . $error);
    }
    if ($status < 200 || $status >= 300) {
        throw new RuntimeException('ACRLA material upload returned HTTP ' . $status . ': ' . $body);
    }

    $decoded = json_decode($body, true);
    if (!is_array($decoded)) {
        throw new RuntimeException('ACRLA material upload returned invalid JSON.');
    }

    $decoded['moodle_filename'] = $filename;
    $decoded['moodle_resource_title'] = $resourcetitle;
    if ($concept) {
        $decoded['concept'] = $concept;
    }
    return $decoded;
}

/**
 * Infer a course-local concept from a Moodle filename/resource name.
 *
 * @param string $text Filename and resource name text.
 * @return string|null Course-local concept.
 */
function local_acrla_infer_concept(string $text): ?string {
    $title = preg_replace('/\.[Pp][Dd][Ff]$/', '', $text);
    $title = str_replace(['_', '-'], ' ', $title);
    $title = preg_replace('/\b(chapter|chap|ch)\s*\d+\b[:.\-\s]*/i', '', $title);
    $title = preg_replace('/\s+/', ' ', trim($title));
    return $title !== '' ? $title : null;
}

/**
 * Send a JSON response and terminate.
 *
 * @param array $payload Response body.
 * @param int $status HTTP status code.
 */
function local_acrla_materials_json(array $payload, int $status = 200): void {
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode($payload);
    die();
}
