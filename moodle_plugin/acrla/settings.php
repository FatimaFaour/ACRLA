<?php
defined('MOODLE_INTERNAL') || die();

if ($hassiteconfig) {
    $settings = new admin_settingpage('local_acrla', get_string('pluginname', 'local_acrla'));

    $settings->add(new admin_setting_configtext(
        'local_acrla/backend_url',
        get_string('backend_url', 'local_acrla'),
        get_string('backend_url_desc', 'local_acrla'),
        'http://localhost:8000',
        PARAM_URL
    ));

    $settings->add(new admin_setting_configcheckbox(
        'local_acrla/enabled',
        get_string('enabled', 'local_acrla'),
        get_string('enabled_desc', 'local_acrla'),
        1
    ));

    $ADMIN->add('localplugins', $settings);
}
