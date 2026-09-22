<?php
defined('MOODLE_INTERNAL') || die();

function xmldb_local_acrla_install() {
    // No additional DB tables needed in Moodle —
    // all data is stored in the ACRLA PostgreSQL backend.
    return true;
}
