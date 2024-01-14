<?php

require_once "robust_lsb.php";
require_once "hamming_code.php";
require_once "utilities.php";

function main(): void
{
    $message = "Secret message";
    $binary_message = message_to_binary($message);
    $hamming_message = hamming_encode($binary_message);
    $image = imagecreatefrompng("images/input.png");
    $image = add_watermark($image, $hamming_message);
    imagepng($image, "images/output.png");

    $image = imagecreatefrompng("images/output.png");
    $hamming_message = read_watermark($image);
    $binary_message = hamming_decode($hamming_message);
    $message = binary_to_message($binary_message);
    echo($message);

    imagedestroy($image);
}

main();
