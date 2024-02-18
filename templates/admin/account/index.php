<?php Template::loadChild('admin/header'); ?>

<div class="content-wrapper">
    <section class="content-header">
        <h1>User Profile </h1>
    </section>

    <section class="content">

        <div class="row">
            <div class="col-md-3">
                <?php if (!$user->isAdmin()): ?>
                <div class="box box-warning text-center" style="background-color: #f39c12; color: #fff; padding: 25px;">
                    <div class="box-header">
                        <i class="fa fa-warning fa-2x" style="font-size: 55px; color: #fff"></i>
                    </div>
                    <div class="box-body box-profile">
                        <h3>Your account is limited. You can request for full access as specified in the documentation</h3>
                        <a href="<?= site_url('documentation/#get_started') ?>" class="btn btn-default" target="_blank"><i class="fa fa-link"></i> See Documentation</a>
                    </div>
                    <!-- /.box-body -->
                </div>
                <?php endif; ?>
                
                <div class="box box-primary">
                    <div class="box-body box-profile">
                        <div class="text-center">
                            <i class="fa fa-user fa-5x"></i>
                        </div>

                        <h3 class="profile-username text-center"><?= h($names) ?></h3>

                        <p class="text-muted text-center"><?= h($affiliation) ?></p>

                        <ul class="list-group list-group-unbordered">
                            <li class="list-group-item">
                                <b>Surveys</b> <a class="pull-right" href="<?= admin_url('survey/list'); ?>"><?= $survey_count ?></a>
                            </li>
                            <li class="list-group-item">
                                <b>Runs(Studies)</b> <a class="pull-right" href="<?= admin_url('run/list'); ?>"><?= $run_count ?></a>
                            </li>
                            <li class="list-group-item">
                                <b>Email Accounts</b> <a class="pull-right" href="<?= admin_url('mail'); ?>"><?= $mail_count ?></a>
                            </li>
                        </ul>

                    </div>
                    <!-- /.box-body -->
                </div>
            </div>

            <div class="col-md-9">

                <?php Template::loadChild('public/alerts'); ?>

                <div class="nav-tabs-custom">
                    <ul class="nav nav-tabs">
                        <li class="active"><a href="#settings" data-toggle="tab" aria-expanded="true">Account Settings</a></li>
                        <li class=""><a href="#api" data-toggle="tab" aria-expanded="false">API Credentials</a></li>
                        <li class=""><a href="#data" data-toggle="tab" aria-expanded="false">Manage collected data</a></li>
                        <li class=""><a href="#moderators" data-toggle="tab" aria-expanded="false">Moderators</a></li>
                    </ul>
                    <div class="tab-content">
                        <div class="tab-pane active" id="settings">
                            <form method="post" action="">
                                <h4 class="lead"> <i class="fa fa-user"></i> Basic Information</h4>

                                <div class="form-group  col-md-6">
                                    <label class="control-label"> First Name </label>
                                    <input class="form-control" name="first_name" value="<?= h($user->first_name) ?>" autocomplete="off">
                                </div>
                                <div class="form-group  col-md-6">
                                    <label class="control-label"> Last Name </label>
                                    <input class="form-control" name="last_name" value="<?= h($user->last_name) ?>" autocomplete="off">
                                </div>
                                <div class="form-group  col-md-12">
                                    <label class="control-label"> Affiliation </label>
                                    <input class="form-control" name="affiliation"  value="<?= h($user->affiliation) ?>" autocomplete="off">
                                </div>
                                <div class="clearfix"></div>

                                <h3 class="lead"> <i class="fa fa-lock"></i> Login Details (changes are effective immediately)</h3>
                                <div class="alert alert-warning col-md-7" style="font-size: 16px;">
                                    <i class="fa fa-warning"></i> &nbsp; If you do not intend to change your password, please leave the password fields empty.
                                </div>
                                <div class="clearfix"></div>
                                
                                <div class="form-group ">
                                    <label class="control-label" for="email"><i class="fa fa-envelope-o fa-fw"></i> New Email</label>
                                    <input class="form-control" type="email" id="email" name="new_email" value="<?= h($user->email) ?>" autocomplete="new-password">
                                </div>
                                
                                <div class="form-group ">
                                    <label class="control-label" for="pass2"><i class="fa fa-key fa-fw"></i> Enter New Password (Choose a secure phrase)</label>
                                    <input class="form-control" type="password" id="pass2" name="new_password" autocomplete="new-password">
                                </div>
                                <div class="form-group ">
                                    <label class="control-label" for="pass3"><i class="fa fa-key fa-fw"></i> Confirm New Password</label>
                                    <input class="form-control" type="password" id="pass3" name="new_password_c" autocomplete="new-password">
                                </div>
                                <p>&nbsp;</p>
                                
                                <div class="col-md-5 no-padding confirm-changes">
                                    <label class="control-label" for="pass"><i class="fa fa-check-circle"></i> Enter Old Password to Save Changes</label>
                                    <div class="input-group input-group">
                                        <input class="form-control" type="password" id="pass" name="password" autocomplete="new-password" placeholder="Old Password">
                                        <span class="input-group-btn">
                                        <button type="submit" class="btn btn-raised btn-primary btn-flat"><i class="fa fa-save"></i> Save Changes</button>
                                        </span>
                                    </div>
                                </div>
                                <div class="clearfix"></div>
                            </form>
                            <form method="post" action="<?= admin_url('account/setupTwoFactor') ?>">
                                <h4 class="lead"> <i class="fa fa-user"></i> Login security</h4>
                                <div class="form-group  col-md-6">
                                    <button type="submit" class="btn btn-raised btn-primary btn-flat"><i class="fa fa-save"></i> Setup 2FA</button>
                                </div>
                                <div class="clearfix"></div>
                            </form>
                        </div>

                        <div class="tab-pane" id="api">
                            <?php if ($api_credentials): ?>
                                <h4 class="lead"> <i class="fa fa-lock"></i> API Credentials</h4>
                                <table class="table table-bordered">
                                    <tr>
                                        <td>Client ID</td>
                                        <td><code><?= $api_credentials['client_id'] ?></code></td>
                                    </tr>

                                    <tr>
                                        <td>Client Secret</td>
                                        <td><code><?= $api_credentials['client_secret'] ?></code></td>
                                    </tr>
                                </table>
                                <p> &nbsp; </p>
                            <?php endif; ?>
                        </div>
                        <div class="tab-pane" id="data">
                            <form method="post" action="">
                                <h4 class="lead"> <i class="fa fa-user"></i> Data management</h4>
                                <label class="control-label" for="pass"><i class="fa fa-check-circle"></i> Delete your account and all associated data</label>
                                <div class="input-group input-group">
                                    <input class="form-control" type="password" id="deleteAcc" name="confirm-delete" autocomplete="new-password" placeholder="Type 'yes' to confirm">
                                    <span class="input-group-btn">
                                      <button type="submit" name="deleteAccBtn" class="btn btn-raised btn-primary btn-flat"><i class="fa fa-save"></i> Delete account</button>
                                    </span>
                                </div>
                                <div class="clearfix"></div>
                            </form>
                        </div>
                        <div class="tab-pane" id="moderators">
                            <?php if(Site::getCurrentUser()->isAdmin()): ?>
                                    <h4 class="lead"> <i class="fa fa-user"></i> Moderators</h4>
                                    <form method="post" action="">
                                        <div class="form-group  col-md-6">
                                            <label class="control-label"> Add moderator (by email)</label>
                                            <input class="form-control" name="add_moderator_name" placeholder="email address">
                                            <input type="submit" class="btn btn-primary" value="Add Moderator">
                                        </div>
                                    </form>
                                    <?php 
                                        foreach(Site::getCurrentUser()->loadModerators() as $mod){
                                            $email = $mod["email"];
                                            echo("
                                            <ul>
                                                <li>
                                                    <i class='fa fa-user'></i> $email
                                                    <form method='POST' action=''>
                                                        <input type='hidden' name='user_delete_email' value='$email'>
                                                        <button type='submit'>Delete</button>
                                                    </form>
                                                </li>
                                            </ul>
                                            ");
                                        }
                                    ?>
                                <div class="clearfix"></div>
                            </div>
                            <?php endif; ?>
                            <?php
                                $modFor = Site::getCurrentUser()->getModeratedAdmin();
                                if($modFor!=null){
                                    echo("<label class='control-label'> You are a moderator for " . $modFor[0]["email"] . ".</label>");
                                }else{
                                    echo("<label class='control-label'> You are not moderating anyone.</label>");
                                }
                            ?>
                            <?php if(Site::getCurrentUser()->getModeratedAdmin()!=null): ?>
                                <li class="dropdown">
                                    <a href="#" class="dropdown-toggle" data-toggle="dropdown"><i class="fa fa-pencil-square"></i> Moderated Surveys <span class="caret"></span></a>
                                    <ul class="dropdown-menu" role="menu">
                                        <?php 
                                            foreach(Site::getCurrentUser()->getStudiesForModerator() as $study){
                                                echo("<li><a href='" . admin_study_url($study["name"]) . "'>" . $study["name"] . "</a></li>");
                                            }
                                        ?>
                                    </ul>
                                </li>
                                <li class="dropdown">
                                    <a href="#" class="dropdown-toggle" data-toggle="dropdown"><i class="fa fa-rocket"></i> Moderated Runs <span class="caret"></span></a>
                                    <ul class="dropdown-menu" role="menu">
                                        <?php 
                                            foreach(Site::getCurrentUser()->getRunsForModerator() as $run){
                                                echo("<li><a href='" . admin_run_url($run["name"]) . "'>" . $run["name"] . "</a></li>");
                                            }
                                        ?>
                                    </ul>
                                </li>
                            <?php endif; ?>
                        <!-- /.tab-pane -->
                    </div>
                    <!-- /.tab-content -->
                </div>

            </div>
        </div>

    </section>


</div>

<?php Template::loadChild('admin/footer'); ?>