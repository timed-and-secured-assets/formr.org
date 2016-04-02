!function(){"use strict";function a(a){this.run=a,this.block=$('<div class="run_unit row"></div>'),a.form.find(".run_units").append(this.block)}function b(b){"undefined"==typeof this.autosaved&&(this.lastSave=$.now(),this.autosaved=!1),this.form=b,this.form.submit(function(){return!1}),this.name=this.form.find(".run_name").val(),this.url=this.form.prop("action"),this.units=[];for(var c=$.parseJSON(this.form.attr("data-units")),d=0;d<c.length;d++)this.units[d]=new a(this),this.loadUnit(c[d],this.units[d]);var e=this;this.form.find("a.add_run_unit").click(function(a){a.preventDefault();var b=$(this).attr("href");return e.addUnit(b),!1}),this.form.find("a.public-toggle").click(this.publicToggle),this.exporter_button=this.form.find("a.export_run_units"),this.exporter_button.click($.proxy(this.exportUnits,this)),this.importer_button=this.form.find("a.import_run_units"),this.importer_button.click($.proxy(this.importUnits,this)),this.reorder_button=this.form.find("a.reorder_units"),this.reorder_button.attr("disabled","disabled").click($.proxy(this.reorderUnits,this)),this.lock_toggle=this.form.find("a.lock-toggle"),this.lock(this.lock_toggle.hasClass("btn-checked"),this.form),this.lock_toggle.click(function(a){a.preventDefault();var b=$(this),c=b.hasClass("btn-checked")?0:1;return b.toggleClass("btn-checked",c),e.lock(!!c,e.form),$.ajax({url:b.attr("href"),dataType:"html",method:"POST",data:{on:c}}).fail(ajaxErrorHandling),!1}),window.onbeforeunload=$.proxy(function(){var a=!1;return $(this.units).each(function(b,c){return c.position_changed||c.unsavedChanges?(a=!0,!1):void 0}),a?"You have unsaved changes.":void 0},this)}a.prototype.init=function(a){this.block.htmlPolyfill($($.parseHTML(a))),this.position=this.block.find(".run_unit_position input.position"),this.position_changed=!1,this.position.change($.proxy(this.position_changes,this)),this.dialog_inputs=this.block.find("div.run_unit_dialog input,div.run_unit_dialog select, div.run_unit_dialog button, div.run_unit_dialog textarea"),this.description=this.block.find(".run_unit_description"),this.unit_id=this.dialog_inputs.filter("input[name=unit_id]").val(),this.run_unit_id=this.dialog_inputs.filter("input[name=run_unit_id]").val(),this.special=this.dialog_inputs.filter("input[name=special]").val(),this.block.attr("id","run_unit_"+this.run_unit_id),this.dialog_inputs.on("input change",$.proxy(this.changes,this)),this.description.on("input change",$.proxy(this.changes,this)),this.save_inputs=this.dialog_inputs.add(this.position).add(this.description),this.block.find(".hastooltip").tooltip({container:"body"}),this.block.find(".select2").select2();var b=this.block.find("input.select2recipient");if(b.length>0&&!b.select2("container").hasClass("select2-container")){var c,d=b.attr("data-select2init");c="object"!=typeof d?$.parseJSON(d):d,b.select2({createSearchChoice:function(a,b){return 0===$(b).filter(function(){return 0===this.text.localeCompare(a)}).length?{id:a,text:a}:void 0},initSelection:function(a,b){var d;d={id:a.val(),text:a.val()},$.each(c,function(b,c){return c.id===a.val()?(d=c,!1):void 0}),b(d)},data:c})}this.unsavedChanges=!1,this.save_button=this.block.find("a.unit_save"),this.block.find("button.from_days").click(function(a){a.preventDefault();var b=$(this).closest(".input-group").find("input[type=number]"),c=b.val();b.val(60*c*24).change()}),this.test_button=this.block.find("a.unit_test"),this.test_button.click($.proxy(this.test,this)),this.remove_button=this.block.find("button.remove_unit_from_run"),this.remove_button.click($.proxy(this.removeFromRun,this)).mouseenter(function(){$(this).addClass("btn-danger")}).mouseleave(function(){$(this).removeClass("btn-danger")});var e=this.block.find("textarea");e[0]&&(this.textarea=$(e[0]),this.session=this.hookAceToTextarea(this.textarea)),e[1]&&(this.textarea2=$(e[1]),this.session2=this.hookAceToTextarea(this.textarea2)),this.run.lock(this.run.lock_toggle.hasClass("btn-checked"),this.block),this.save_button.attr("disabled",!0).removeClass("btn-info").text("Saved").click($.proxy(this.save,this))},a.prototype.position_changes=function(a){this.position_changed||(this.position_changed=!0,this.run.reorder_button.addClass("btn-info").removeAttr("disabled")),this.position.parent().addClass("pos_changed")},a.prototype.changes=function(a){this.unsavedChanges||(this.unsavedChanges=!0,this.save_button.addClass("btn-info").removeAttr("disabled").text("Unsaved changes…"),this.test_button.attr("disabled","disabled"))},a.prototype.test=function(a){a.preventDefault();var b=this.test_button.text();this.test_button.attr("disabled",!0).html(b+bootstrap_spinner());this.block;return $.ajax({url:this.run.url+"/"+this.test_button.attr("href"),dataType:"html",data:{run_unit_id:this.run_unit_id,special:this.special},method:"GET"}).done($.proxy(function(a){var c=bootstrap_modal("Test Results",a);$(".opencpu_accordion",c).collapse({toggle:!0}),this.test_button.html(b).removeAttr("disabled");var d=c.find("pre code");Array.prototype.forEach.call(d,hljs.highlightBlock),c.find(".download_r_code").length>0&&c.find(".download_r_code").click(download_next_textarea)},this)).fail($.proxy(function(a,c,d,e){this.test_button.attr("disabled",!1).html(b),ajaxErrorHandling(a,c,d,e)},this)),!1},a.prototype.save=function(a){a.preventDefault();var b=this.save_button.text();this.save_button.attr("disabled","disabled").html(b+bootstrap_spinner()),this.session&&this.textarea.val(this.session.getValue()),this.session2&&this.textarea2.val(this.session2.getValue());this.block;return $.ajax({url:this.run.url+"/"+this.save_button.attr("href"),dataType:"html",data:this.save_inputs.serialize(),method:"POST"}).done($.proxy(function(a){""!==a?$.proxy(this.init(a),this):(this.save_button.attr("disabled",!0).removeClass("btn-info").text("Saved").click($.proxy(this.save,this)),this.unsavedChanges=!1,this.test_button.removeAttr("disabled"))},this)).fail($.proxy(function(a,c,d,e){this.save_button.removeAttr("disabled").html(b),ajaxErrorHandling(a,c,d,e)},this)),!1},a.prototype.hookAceToTextarea=function(a){var b=a.data("editor"),c=$("<div>",{position:"absolute",width:a.width(),height:a.height(),"class":a.attr("class")}).insertBefore(a);a.css("display","none"),this.editor=ace.edit(c[0]),this.editor.setOptions({minLines:a.attr("rows")?a.attr("rows"):3,maxLines:30}),this.editor.setTheme("ace/theme/textmate"),this.editor.$blockScrolling=1/0;var d=this.editor.getSession();return d.setValue(a.val()),this.editor.renderer.setShowGutter(!1),d.setUseWrapMode(!0),d.setWrapLimitRange(42,42),d.setMode("ace/mode/"+b),this.editor.on("change",$.proxy(this.changes,this)),d},a.prototype.removeFromRun=function(a){a.preventDefault(),$(".tooltip").hide();var b=this.block;return b.hide(),$.ajax({url:this.run.url+"/"+this.remove_button.attr("href"),dataType:"html",data:{run_unit_id:this.run_unit_id},method:"POST"}).done($.proxy(function(a){b.html(a),b.show();var c=this.run.units.indexOf(this);c>-1&&this.run.units.splice(c,1)},this)).fail(function(a,c,d,e){b.show(),ajaxErrorHandling(a,c,d,e)}),!1},a.prototype.serialize=function(){var a=this.save_inputs.serializeArray(),b={};b.type=this.block.find(".run_unit_inner").data("type");for(var c=0;c<a.length;c++)"unit_id"!=a[c].name&&"run_unit_id"!=a[c].name&&"position"!=a[c].name.substr(0,8)?b[a[c].name]=a[c].value:"position"==a[c].name.substr(0,8)&&(b.position=a[c].value);return b},b.prototype.getMaxPosition=function(){var a=null;return $(this.units).each(function(b,c){var d=+c.position.val();null===a?a=d:d>a&&(a=d)}),null===a&&(a=0),a},b.prototype.loadUnit=function(a,b){$.ajax({url:this.url+"/ajax_get_unit",data:a,dataType:"html",success:$.proxy(function(a,c){b.init(a)},this)})},b.prototype.addUnit=function(b){var c=this.getMaxPosition(),d=new a(this);this.units.push(d),$.ajax({url:b,dataType:"html",method:"POST",data:{position:c+10}}).done($.proxy(function(a){d.init(a)},this)).fail(ajaxErrorHandling)},b.prototype.exportUnits=function(){var a={},b=this.url,c=!1,d=$("<div />"),e=this.lock_toggle.hasClass("btn-checked");e&&this.lock(!1,this.form);for(var f=0;f<this.units.length;f++){var g=this.units[f].serialize();g.unit_id=this.units[f].unit_id,g.run_unit_id=this.units[f].run_unit_id,c=c||this.units[f].unsavedChanges,a[g.position]=g,d.append($($.parseHTML(getHTMLTemplate("tpl-export-unit-block",{unit_pos:g.position,unit_json:JSON.stringify(g,null,"	")}))))}if(e&&this.lock(e,this.form),c)return void bootstrap_modal("Please save all changes before export.","Unsaved Changes");var h=d.html(),i=$($.parseHTML(getHTMLTemplate("tpl-export-units",{run_name:this.name,export_html:h})));i.find("form#export_run_units").attr("action",b+"/export"),i.on("shown.bs.modal",function(){i.find(".confirm-export").click(function(b){var c=$.trim(i.find("input[name=export_name]").val()),d=/^[a-z0-9_\s]+$/i;if(!c||!d.test(c))return bootstrap_alert("Enter a valid export name","Export name invalid.",".run_export_before_alert"),!1;var e={},f=i.find(".run-export-unit-block");return f.each(function(){var b=$(this).find(".select"),c=parseInt(b.data("selected"),10),d=parseInt(b.data("position"),10);!c||isNaN(c)||isNaN(d)||(e[d]=a[d])}),$.isEmptyObject(e)?(bootstrap_alert("You need to select at least one unit to export.","Nothing chosen.",".run_export_before_alert"),!1):(i.find("input[name=units]").val(JSON.stringify(e)),window.setTimeout(function(){i.find(".cancel-export").trigger("click")},100),!0)})}).on("hidden.bs.modal",function(){i.remove()}).modal("show");var j=i.find("pre code");j.each(function(){var a=$(this);hljs.highlightBlock(a.get(0)),a.parents(".run-export-unit-block").find(".select").on("click",function(){var a=$(this),b=parseInt(a.data("selected"),10);b?(a.data("selected",0),a.find("i").removeClass("fa-check")):(a.data("selected",1),a.find("i").addClass("fa-check"))})})},b.prototype.importUnits=function(){var a=this,b=$("#run-import-modal-dialog");return b.length?b.modal("show"):void $.get(this.url+"/ajax_run_import",{dialog:!0},function(c){b=$($.parseHTML(getHTMLTemplate("tpl-import-units",{content:c}))).attr("id","run-import-modal-dialog"),b.find("select").bind("change",function(){var a=parseInt($(this).val(),10);if(!isNaN(a)){var c="selected-run-export-"+a,d=getHTMLTemplate(c);b.find("textarea").val(JSON.stringify($.parseJSON(d),null,"	"))}}),b.on("shown.bs.modal",function(){b.find(".confirm-import").click(function(b){b.preventDefault(),$(this).html(bootstrap_spinner());var c=$(this).parents("form");return c.find("input[name=position]").val(a.getMaxPosition()+10),c.submit(),!0})}).on("hidden.bs.modal",function(){b.remove()}).modal("show")})},b.prototype.reorderUnits=function(a){if(a.preventDefault(),"undefined"==typeof this.reorder_button.attr("disabled")){var b,c={},d=[],e=!1;if($(this.units).each(function(a,f){b=+f.position.val(),$.inArray(b,d)>-1?(bootstrap_alert("You used the position "+b+" more than once, therefore the new order could not be saved. <a href='#run_unit_"+f.run_unit_id+"'>Click here to scroll to the duplicated position.</a>","Error.",".run_units"),e=!0):(c[f.run_unit_id]=b,d.push(b))}),!e)return $.ajax({url:this.reorder_button.attr("href"),dataType:"html",method:"POST",data:{position:c}}).done($.proxy(function(a){$(this.units).each(function(a,b){b.position_changed=!1}),this.reorder_button.removeClass("btn-info").attr("disabled","disabled");var b=d.join(","),c=d.sort(function(a,b){return a-b}).join(",");if(this.form.find(".pos_changed").removeClass("pos_changed"),b!=c){var e=this.form;$(this.units.sort(function(a,b){return+a.position.val()-+b.position.val()})).each(function(a,b){e.find(".run_units").append(b.block)})}},this)).fail(ajaxErrorHandling),!1}},b.prototype.lock=function(a,b){b.find(".import_run_units, .run_unit_description, .position, .remove_unit_from_run, .reorder_units, .unit_save, .form-control, select, .from_days, .add_run_unit").each(function(b,c){a?(c.onclick&&(c.onclick_disabled=c.onclick,c.onclick=function(a){return a.preventDefault(),!1}),$(c).attr("data-old_disabled",$(c).attr("disabled")),$(c).attr("disabled","disabled")):(c.onclick_disabled&&(c.onclick=c.onclick_disabled),$(c).attr("data-old-disabled")&&""!==$(c).attr("data-old-disabled")?$(c).attr("disabled",$(c).attr("data-old-disabled")):$(c).removeAttr("disabled"))})},b.prototype.publicToggle=function(a){var b=$(this);return b.parents(".btn-group").find(".btn-checked").removeClass("btn-checked"),b.toggleClass("btn-checked",1),$.ajax({url:b.attr("href"),dataType:"html",method:"POST"}).fail(ajaxErrorHandling),!1},$(document).ready(function(){$(".edit_run").each(function(a,c){new b($(c))})})}();