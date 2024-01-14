<?php

function add_watermark($image, $binary_message)
{
    $binary_message = $binary_message . "0000000000000000";

    $k = 5;

    $x = 0;
    $y = 0;

    for ($i = 0; $i < strlen($binary_message); $i++) {
        for ($dx = 0; $dx < $k; $dx++) {
            for ($dy = 0; $dy < $k; $dy++) {
                $color_index = imagecolorat($image, $x + $dx, $y + $dy);
                $colors = imagecolorsforindex($image, $color_index);

                $colors["red"] = ($colors["red"] & 0xFE) | ($binary_message[$i] == "1" ? 1 : 0);
                $colors["green"] = ($colors["green"] & 0xFE) | ($binary_message[$i] == "1" ? 1 : 0);
                $colors["blue"] = ($colors["blue"] & 0xFE) | ($binary_message[$i] == "1" ? 1 : 0);

                $new_color = imagecolorallocate($image, $colors["red"], $colors["green"], $colors["blue"]);
                imagesetpixel($image, $x + $dx, $y + $dy, $new_color);
            }
        }

        $x += $k;
        if ($x >= imagesx($image)) {
            $x = 0;
            $y += $k;
        }
    }

    return $image;
}

function read_watermark($image): string
{
    $binary_message = "";

    $k = 5;

    $x = 0;
    $y = 0;

    while (true) {
        $bit_array = array("red" => array(), "green" => array(), "blue" => array());

        for ($dx = 0; $dx < $k; $dx++) {
            for ($dy = 0; $dy < $k; $dy++) {
                $color_index = imagecolorat($image, $x + $dx, $y + $dy);
                $colors = imagecolorsforindex($image, $color_index);

                $bit_array["red"][$dx * $k + $dy] = ($colors["red"] & 1);
                $bit_array["green"][$dx * $k + $dy] = ($colors["green"] & 1);
                $bit_array["blue"][$dx * $k + $dy] = ($colors["blue"] & 1);
            }
        }

        $bit_array["red"] = array_sum($bit_array["red"]) >= ceil($k * $k / 2) ? 1 : 0;
        $bit_array["green"] = array_sum($bit_array["green"]) >= ceil($k * $k / 2) ? 1 : 0;
        $bit_array["blue"] = array_sum($bit_array["blue"]) >= ceil($k * $k / 2) ? 1 : 0;

        $bit = array_sum($bit_array) >= 2 ? 1 : 0;
        $binary_message .= $bit;

        if (strlen($binary_message) >= 16 && str_ends_with($binary_message, "0000000000000000")) {
            $binary_message = substr($binary_message, 0, -16);
            break;
        }

        $x += $k;
        if ($x >= imagesx($image)) {
            $x = 0;
            $y += $k;
            if ($y >= imagesy($image)) {
                break;
            }
        }
    }

    return $binary_message;
}
