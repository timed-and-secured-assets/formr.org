<?php

function hamming_7_4_encode($data): array
{
    $p1 = ($data[0] + $data[1] + $data[3]) % 2;
    $p2 = ($data[0] + $data[2] + $data[3]) % 2;
    $p3 = ($data[1] + $data[2] + $data[3]) % 2;

    return [$p1, $p2, $data[0], $p3, $data[1], $data[2], $data[3]];
}

function hamming_7_4_decode($data): array
{
    $p1 = ($data[0] + $data[2] + $data[4] + $data[6]) % 2;
    $p2 = ($data[1] + $data[2] + $data[5] + $data[6]) % 2;
    $p3 = ($data[3] + $data[4] + $data[5] + $data[6]) % 2;

    $errorPos = $p1 * 1 + $p2 * 2 + $p3 * 4;
    if ($errorPos != 0) {
        $data[$errorPos - 1] = ($data[$errorPos - 1] == 1) ? 0 : 1;
    }

    return [$data[2], $data[4], $data[5], $data[6]];
}

function hamming_encode($binary_message): string
{
    $encoded_message = "";

    for ($i = 0; $i < strlen($binary_message); $i += 4) {
        $data = array_map("intval", str_split(substr($binary_message, $i, 4)));
        $data = hamming_7_4_encode($data);
        $encoded_message .= implode("", $data);
    }

    return $encoded_message;
}

function hamming_decode($binary_message): string
{
    $decoded_message = "";

    for ($i = 0; $i < strlen($binary_message); $i += 7) {
        $data = array_map("intval", str_split(substr($binary_message, $i, 7)));
        $data = hamming_7_4_decode($data);
        $decoded_message .= implode("", $data);
    }

    return $decoded_message;
}
