$(function(){"use strict";function a(a,b){$(b).click(function(a){a.preventDefault();var b=$(this),c=b.attr("href");return""===c?!1:(b.attr("href",""),$.ajax({type:"GET",url:c,dataType:"html"}).done($.proxy(function(a){var b=$(this);b.attr("href",c),b.hasClass("danger")||b.css("color","green");var d=b.find("i.fa");d.hasClass("fa-stethoscope")?(d.addClass("fa-heartbeat"),d.removeClass("fa-stethoscope")):d.hasClass("fa-heartbeat")?(d.removeClass("fa-heartbeat"),d.addClass("fa-stethoscope")):bootstrap_modal("Alert",a,"tpl-feedback-modal"),b.hasClass("refresh_on_success")&&document.location.reload(!0)},this)).fail($.proxy(function(a,b,d,e){$(this).attr("href",c),ajaxErrorHandling(a,b,d,e)},this)),!1)})}function b(a,b){$(b).submit(function(a){a.preventDefault();var b=$(this),c=b.find("button[type=submit].btn");return c.attr("disabled",!0),$.ajax({type:b.attr("method"),url:b.attr("action"),data:b.serialize(),dataType:"html"}).done($.proxy(function(a){c.attr("disabled",!1),c.css("color","green"),$(".main_body").prepend(a),c.hasClass("refresh_on_success")&&document.location.reload(!0)},this)).fail($.proxy(function(a,b,d,e){c.attr("disabled",!1),ajaxErrorHandling(a,b,d,e)},this)),!1})}function c(a){var b=parseInt($(this).data("user"),10),c=$(this).data("email");if(b&&c){var f={user_id:b,user_email:c,user_api:!0,api_action:"get"};e(saAjaxUrl,f,function(a){a&&a.success&&d(a.data,f)})}}function d(a,b){var c=$($.parseHTML(getHTMLTemplate("tpl-user-api",{user:" ("+a.user+")",client_id:a.client_id,client_secret:a.client_secret})));a.client_id?c.find(".api-create").remove():c.find(".api-change, .api-delete").remove(),c.on("shown.bs.modal",function(){c.find(".api-create").click(function(){if(confirm("Are you sure?")){var a={user_id:b.user_id,user_email:b.user_email,user_api:!0,api_action:"create"};e(saAjaxUrl,a,function(a){a&&a.success&&(c.modal("hide"),d(a.data,b))})}}),c.find(".api-change").click(function(){if(confirm("Are you sure?")){var a={user_id:b.user_id,user_email:b.user_email,user_api:!0,api_action:"change"};e(saAjaxUrl,a,function(a){a&&a.success&&(c.modal("hide"),d(a.data,b))})}}),c.find(".api-delete").click(function(){if(confirm("Are you sure?")){var f={user_id:b.user_id,user_email:b.user_email,user_api:!0,api_action:"delete"};e(saAjaxUrl,f,function(e){e&&e.success&&(c.modal("hide"),d({user:a.user,client_id:"",client_secret:""},b))})}})}).on("hidden.bs.modal",function(){c.remove()}).modal("show")}function e(a,b,c,d){d=d||"json",$.ajax({type:"POST",url:a,data:b,dataType:d,success:function(a,b,d){c(a)},error:function(a,b){$(".main_body").prepend(b)},beforeSend:function(a){}})}function f(){var a=$(this);"single"===a.data("active")?(a.siblings(".single").addClass("hidden"),a.siblings(".multiple").removeClass("hidden"),a.data("active","multiple")):(a.siblings(".single").removeClass("hidden"),a.siblings(".multiple").addClass("hidden"),a.data("active","single"))}var g;$(".form-ajax").each(b),$(".link-ajax").each(a),$(".api-btn").click(c),$(".sessions-search-switch").click(f),$(".hidden_debug_message").length>0&&($(".show_hidden_debugging_messages").click(function(){return $(".hidden_debug_message").toggleClass("hidden"),!1}),$(".show_hidden_debugging_messages").attr("disabled",!1)),$("abbr.abbreviated_session").click(function(){$(this).text()!==$(this).data("full-session")?$(this).text($(this).data("full-session")):$(this).text($(this).data("full-session").substr(0,10)+"…")}),$(".download_r_code").length>0&&$(".download_r_code").click(download_next_textarea),$("#confirm-delete").on("show.bs.modal",function(b){g=$(b.relatedTarget);var c=$(this);g.parents("tr").css("background-color","#ee5f5b"),$(this).find(".danger").attr("href",g.data("href")),a(1,$(this).find(".danger")),$(this).find(".danger").click(function(a){g.css("color","#ee5f5b"),c.hasClass("refresh_on_success")&&window.setTimeout(function(){document.location.reload(!0)},200),c.modal("hide")})}).on("hide.bs.modal",function(a){g.parents("tr").css("background-color","transparent")})});