<?php

class View {
    
    protected $template = null;
    
    protected $variables = array();

    public function __construct($template = null, $variables = array()) {
        $this->template = $template;
        $this->variables = $variables;
    }

    public function setTemplate($template) {
        if (!is_string($template)) {
            throw new Exception('Invalid template name');
        }
        
        $this->template = $template;
    }

    public function setVariables(array $variables) {
        $this->variables = $variables;
    }

    public function render() {
        return Template::get($this->template, $this->variables);
    }
}