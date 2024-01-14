<?php

function message_to_binary($message): string
{
    $binary_message = "";

    for ($i = 0; $i < strlen($message); $i++) {
        $binary_message .= str_pad(decbin(ord($message[$i])), 8, "0", STR_PAD_LEFT);
    }

    return $binary_message;
}

function binary_to_message($binary_message): string
{
    $message = "";

    for ($i = 0; $i < strlen($binary_message); $i += 8) {
        $message .= chr(bindec(substr($binary_message, $i, 8)));
    }

    return $message;
}
