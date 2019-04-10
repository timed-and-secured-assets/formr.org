<?php

/**
 * Group Admin Ajax requests here
 * Takes in a controller of type AdminController
 * which already have most required 'global' variables defined
 * 
 */
class AdminAjaxController {

    /**
     *
     * @var AdminController
     */
    protected $controller;

    /**
     *
     * @var Site
     */
    protected $site;

    /**
     * @var Request
     */
    protected $request;

    /**
     *
     * @var Response
     */
    protected $response;

    /**
     *
     * @var DB
     */
    protected $dbh;

    public function __construct(AdminController $controller) {
        $this->controller = $controller;
        $this->site = $controller->getSite();
        $this->dbh = $controller->getDB();
        $this->request = new Request();
        $this->response = new Response();
    }

    /**
     * Execute the corresponding ajax method
     * 
     * @param string $method
     * @param AdminController $controller
     *
     * @return Response
     */
    public static function call($method, AdminController $controller) {
        $self = new self($controller);
        $action = $self->getPrivateAction($method);
        $self->$action();
        return $self->response;
    }

    private function ajaxCreateRunUnit() {
        if (!Request::isAjaxRequest()) {
            formr_error(406, 'Not Acceptable');
        }

        $unit = $this->controller->createRunUnit();
        if ($unit->valid) {
            $unit->addToRun($this->controller->run->id, $unit->position);
            alert('<strong>Success.</strong> ' . ucfirst($unit->type) . ' unit was created.', 'alert-success');
            $content = $unit->displayForRun($this->site->renderAlerts());
        } else {
            $this->response->setStatusCode(500, 'Bad Request');
            $msg = '<strong>Sorry.</strong> Run unit could not be created.';
            $msg .= !empty($unit) ? implode("\n", $unit->errors) : '';
            alert($msg, 'alert-danger');
            $content = $this->site->renderAlerts();
        }

        return $this->response->setContent($content);
    }

    private function ajaxGetUnit() {
        if (!Request::isAjaxRequest()) {
            formr_error(406, 'Not Acceptable');
        }

        $run = $this->controller->run;
        $dbh = $this->dbh;

        if ($run_unit_id = $this->request->getParam('run_unit_id')) {
            $special = $this->request->getParam('special');

            $unit_info = $run->getUnitAdmin($run_unit_id, $special);
            $unit_factory = new RunUnitFactory();
            $unit = $unit_factory->make($dbh, null, $unit_info, null, $run);

            $content = $unit->displayForRun();
        } else {
            $this->response->setStatusCode(500, 'Bad Request');
            $msg = '<strong>Sorry.</strong> Missing run unit.';
            $msg .= !empty($unit) ? implode("\n", $unit->errors) : '';
            alert($msg, 'alert-danger');
            $content = $this->site->renderAlerts();
        }

        $this->response->setContent($content);
    }

    private function ajaxRemind() {
        if ($this->request->bool('get_count') === true) {
            $sessions = $this->getSessionRemindersSent($this->request->int('run_session_id'));
            $count = array();
            foreach ($sessions as $sess) {
                if (!isset($count[$sess['unit_id']])) {
                    $count[$sess['unit_id']] = 0;
                }
				$count[$sess['unit_id']]++;
            }

            $this->response->setContentType('application/json');
            return $this->response->setJsonContent($count);
        }

        $run = $this->controller->run;
        // find the last email unit
        $email = $run->getReminder($this->request->getParam('reminder'), $this->request->getParam('session'), $this->request->getParam('run_session_id'));
        $email->run_session = new RunSession($this->dbh, $run->id, null, $this->request->getParam('session'), $run);
        if ($email->exec() !== false) {
            alert('<strong>Something went wrong with the reminder.</strong> Run: ' . $run->name, 'alert-danger');
        } else {
            alert('Reminder sent!', 'alert-success');
        }
        $email->end();

        if (Request::isAjaxRequest()) {
            $content = $this->site->renderAlerts();
            return $this->response->setContent($content);
        } else {
            $this->request->redirect(admin_run_url($run->name, 'user_overview'));
        }
    }

    private function ajaxToggleTesting() {
        $run = $this->controller->run;
        $dbh = $this->dbh;

        $run_session = new RunSession($dbh, $run->id, null, $this->request->getParam('session'), $run);

        $status = $this->request->getParam('toggle_on') ? 1 : 0;
        $run_session->setTestingStatus($status);

        if (Request::isAjaxRequest()) {
            $content = $this->site->renderAlerts();
            return $this->response->setContent($content);
        } else {
            $this->request->redirect(admin_run_url($run->name, 'user_overview'));
        }
    }

    private function ajaxSendToPosition() {
        $run = $this->controller->run;
        $dbh = $this->dbh;

        $run_session = new RunSession($dbh, $run->id, null, $this->request->str('session'), $run);
        $new_position = $this->request->int('new_position');

        if (!$run_session->forceTo($new_position)) {
            alert('<strong>Something went wrong with the position change.</strong> Run: ' . $run->name, 'alert-danger');
            $this->response->setStatusCode(500, 'Bad Request');
        }

        if (Request::isAjaxRequest()) {
            $content = $this->site->renderAlerts();
            return $this->response->setContent($content);
        } else {
            $this->request->redirect(admin_run_url($run->name, 'user_overview'));
        }
    }

    private function ajaxNextInRun() {
        $run = $this->controller->run;
        $dbh = $this->dbh;

        $run_session = new RunSession($dbh, $run->id, null, $_GET['session'], $run);

        if (!$run_session->endUnitSession()) {
            alert('<strong>Something went wrong with the unpause.</strong> in run ' . $run->name, 'alert-danger');
            $this->response->setStatusCode(500, 'Bad Request');
        }

        if (Request::isAjaxRequest()) {
            $content = $this->site->renderAlerts();
            return $this->response->setContent($content);
        } else {
            $this->request->redirect(admin_run_url($run->name, 'user_overview'));
        }
    }

    private function ajaxSnipUnitSession() {
        $run = $this->controller->run;
        $dbh = $this->dbh;
        $run_session = new RunSession($dbh, $run->id, null, $this->request->getParam('session'), $run);

        $unit_session = $run_session->getUnitSession();
        if ($unit_session) {
            $deleted = $dbh->delete('survey_unit_sessions', array('id' => $unit_session->id));
            if ($deleted) {
                alert('<strong>Success.</strong> You deleted the data at the current position.', 'alert-success');
            } else {
                alert('<strong>Couldn\'t delete.</strong>', 'alert-danger');
                $this->response->setStatusCode(500, 'Bad Request');
            }
        } else {
            alert("No unit session found", 'alert-danger');
        }

        if (Request::isAjaxRequest()) {
            $content = $this->site->renderAlerts();
            return $this->response->setContent($content);;
        } else {
            $this->request->redirect(admin_run_url($run->name, 'user_overview'));
        }
    }

    private function ajaxDeleteUser() {
        $run = $this->controller->run;
        $deleted = $this->dbh->delete('survey_run_sessions', array('id' => $this->request->getParam('run_session_id'), 'run_id' => $run->id));
        if ($deleted) {
            alert('User with session ' . h($_GET['session']) . ' was deleted.', 'alert-info');
        } else {
            alert('User with session ' . h($_GET['session']) . ' could not be deleted.', 'alert-warning');
            $this->response->setStatusCode(500, 'Bad Request');
        }

        if (Request::isAjaxRequest()) {
            $content = $this->site->renderAlerts();
            return $this->response->setContent($content);
        } else {
            $this->request->redirect(admin_run_url($run->name, 'user_overview'));
        }
    }

    private function ajaxDeleteUnitSession() {
        $run = $this->controller->run;
        $deleted = $this->dbh->delete('survey_unit_sessions', array('id' => $this->request->int('session_id')));

        if ($deleted) {
            alert('<strong>Success.</strong> You deleted this unit session.', 'alert-success');
        } else {
            alert('<strong>Couldn\'t delete.</strong> Sorry. <pre>' . print_r($del->errorInfo(), true) . '</pre>', 'alert-danger');
            $this->response->setStatusCode(500, 'Bad Request');
        }

        if (Request::isAjaxRequest()) {
            $content = $this->site->renderAlerts();
            return $this->response->setContent($content);;
        } else {
            $this->request->redirect(admin_run_url($run->name, 'user_detail'));
        }
    }

    private function ajaxRemoveRunUnitFromRun() {
        if (!Request::isAjaxRequest()) {
            formr_error(406, 'Not Acceptable');
        }

        $run = $this->controller->run;
        $dbh = $this->dbh;

        if (($run_unit_id = $this->request->getParam('run_unit_id'))) {
            $special = $this->request->getParam('special');

            $unit_info = $run->getUnitAdmin($run_unit_id, $special);
            $unit_factory = new RunUnitFactory();
            /* @var $unit RunUnit */
            $unit = $unit_factory->make($dbh, null, $unit_info, null, $run);
            if (!$unit) {
                formr_error(404, 'Not Found', 'Requested Run Unit was not found');
            }
            $sess_key = __METHOD__ . $unit->id;
            $results = $unit->howManyReachedItNumbers();
            $has_sessions = $results && (array_val($results, 'begun') || array_val($results, 'finished') || array_val($results, 'expired'));

            if ($has_sessions && !Session::get($sess_key)) {
                Session::set($sess_key, $unit->id);
                $content = 'warn';
                return $this->response->setContent($content);
            } elseif (!$has_sessions || (Session::get($sess_key) === $unit->id && $this->request->getParam('confirm') === 'yes')) {
                if ($unit->removeFromRun($special)) {
                    alert('<strong>Success.</strong> Unit with ID ' . $this->request->run_unit_id . ' was deleted.', 'alert-success');
                } else {
                    $this->response->setStatusCode(500, 'Bad Request');
                    $alert_msg = '<strong>Sorry, could not remove unit.</strong> ';
                    $alert_msg .= implode($unit->errors);
                    alert($alert_msg, 'alert-danger');
                }
            }
        }

        Session::delete($sess_key);
        $content = $this->site->renderAlerts();
        return $this->response->setContent($content);
    }

    private function ajaxReorder() {
        if (!Request::isAjaxRequest()) {
            formr_error(406, 'Not Acceptable');
        }

        $run = $this->controller->run;
        $positions = $this->request->arr('position');
        if ($positions) {
            $unit = $run->reorder($positions);
            $content = '';
        } else {
            $this->response->setStatusCode(500, 'Bad Request');
            $msg = '<strong>Sorry.</strong> Re-ordering run units failed.';
            $msg .= !empty($unit) ? implode("\n", $unit->errors) : '';
            alert($msg, 'alert-danger');
            $content = $this->site->renderAlerts();
        }

        return $this->response->setContent($content);
    }

    private function ajaxRunImport() {
        $run = $this->controller->run;
        $site = $this->site;

        if (!Request::isAjaxRequest()) {
            formr_error(406, 'Not Acceptable');
        }

        // If only showing dialog then show it and exit
        $dialog_only = $site->request->bool('dialog');
        if ($dialog_only) {
            // Read on exported runs from configured directory
            $dir = Config::get('run_exports_dir');
            if (!($exports = (array) get_run_dir_contents($dir))) {
                $exports = array();
            }

            $view = new View('admin/run/run_import_dialog', array(
                'exports' => $exports,
                'run' => $this->controller->run
            ));
            return $this->response->setContent($view->render());
        }
    }

    private function ajaxRunLockedToggle() {
        if (!Request::isAjaxRequest()) {
            formr_error(406, 'Not Acceptable');
        }
        $run = $this->controller->run;
        $lock = $this->request->int('on');
        if (in_array($lock, array(0, 1))) {
            return $run->toggleLocked($lock);
        }
    }

    private function ajaxRunPublicToggle() {
        if (!Request::isAjaxRequest()) {
            formr_error(406, 'Not Acceptable');
        }
        $run = $this->controller->run;
        $pub = $this->request->int('public');
        if (!$run->togglePublic($pub)) {
            $this->response->setStatusCode(500, 'Bad Request');
        }
    }

    private function ajaxSaveRunUnit() {
        if (!Request::isAjaxRequest()) {
            formr_error(406, 'Not Acceptable');
        }

        $run = $this->controller->run;
        $dbh = $this->dbh;
        $content = '';

        $unit_factory = new RunUnitFactory();
        if ($run_unit_id = $this->request->getParam('run_unit_id')) {
            $special = $this->request->getParam('special');
            $unit_info = $run->getUnitAdmin($run_unit_id, $special);

            $unit = $unit_factory->make($dbh, null, $unit_info, null, $run);
            $unit->create($_POST);
            if ($unit->valid && ($unit->hadMajorChanges() || !empty($this->site->alerts))) {
                $content = $unit->displayForRun($this->site->renderAlerts());
            }
        } else {
            $this->response->setStatusCode(500, 'Bad Request');
            $alert_msg = "<strong>Sorry.</strong> Something went wrong while saving. Please contact formr devs, if this problem persists.";
            if (!empty($unit)) {
                $alert_msg .= implode("\n", $unit->errors);
            }
            alert($alert_msg, 'alert-danger');
            $content = $this->site->renderAlerts();
        }

        return $this->response->setContent($content);
    }

    private function ajaxSaveSettings() {
        if (!Request::isAjaxRequest()) {
            formr_error(406, 'Not Acceptable');
        }

        $run = $this->controller->run;
        $post = new Request($_POST);
        if ($run->saveSettings($post->getParams())) {
            alert('Settings saved', 'alert-success');
        } else {
            $this->response->setStatusCode(500, 'Bad Request');
            alert('<strong>Error.</strong> ' . implode(".\n", $run->errors), 'alert-danger');
        }

        $content = $this->site->renderAlerts();
        return $this->response->setContent($content);
    }

    private function ajaxTestUnit() {
        if (!Request::isAjaxRequest()) {
            formr_error(406, 'Not Acceptable');
        }

        $run = new Run($this->dbh, $this->controller->run->name);

        if ($run_unit_id = $this->request->getParam('run_unit_id')) {
            $special = $this->request->getParam('special');
            $unit = $run->getUnitAdmin($run_unit_id, $special);
            $unit_factory = new RunUnitFactory();
            $unit = $unit_factory->make($this->dbh, null, $unit, null, $run);
            $test_content = $unit->test();
            $content = $this->site->renderAlerts();
            $content .= $test_content;
        } else {
            $this->response->setStatusCode(500, 'Bad Request');
            $alert_msg = "<strong>Sorry.</strong> An error occured during the test.";
            $alert_msg .= isset($unit) ? implode("\n", $unit->errors) : '';
            alert($alert_msg, 'alert-danger');
            $content = $this->site->renderAlerts();
        }

        return $this->response->setContent($content);
    }

    private function ajaxUserBulkActions() {
        if (!Request::isAjaxRequest()) {
            formr_error(406, 'Not Acceptable');
        }

        $action = $this->request->str('action');
        $sessions = $this->request->arr('sessions');
        $qs = $res = array();
        if (!$action || !$sessions) {
            $this->response->setStatusCode(500, 'Bad Request');
            return $this->response->setContent('Missing Parameters');
        }

        if ($action === 'toggleTest') {
            $count = RunSession::toggleTestingStatus($sessions);
            alert("{$count} selected session(s) were successfully modified", 'alert-success');
            $res['success'] = true;
        } elseif ($action === 'sendReminder') {
            $run = $this->controller->run;
            $count = 0;
            foreach ($sessions as $sess) {
                $runSession = new RunSession($this->dbh, $run->id, null, $sess, $run);
                $email = $run->getReminder($this->request->int('reminder'), $sess, $runSession->id);
                $email->run_session = $runSession;
                if ($email->exec() === false) {
                    $count++;
                }
                $email->end();
            }

            if ($count) {
                alert("{$count} session(s) have been sent the reminder '{$email->getSubject()}'", 'alert-success');
                $res['success'] = true;
            } else {
                $res['error'] = $this->site->renderAlerts();
            }
        } elseif ($action === 'deleteSessions') {
            $count = RunSession::deleteSessions($sessions);
            alert("{$count} selected session(s) were successfully deleted", 'alert-success');
            $res['success'] = true;
        } elseif ($action === 'positionSessions') {
            $count = RunSession::positionSessions($sessions, $this->request->int('pos'));
            alert("{$count} selected session(s) were successfully moved", 'alert-success');
            $res['success'] = true;
        }

        $this->response->setContentType('application/json');
        return $this->response->setJsonContent($res);
    }

    protected function getPrivateAction($name) {
        $parts = array_filter(explode('_', $name));
        $action = array_shift($parts);
        $class = __CLASS__;
        foreach ($parts as $part) {
            $action .= ucwords(strtolower($part));
        }
        if (!method_exists($this, $action)) {
            throw new Exception("Action '$name' is not found in $class.");
        }
        return $action;
    }

    protected function getSessionRemindersSent($run_session_id) {
        return RunSession::getSentRemindersBySessionId($run_session_id);
    }

}
