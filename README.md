# Rummikub
* [Mouse Control](https://github.com/RobRoyce/fpga_mouse_controller_basys3)

---

# Progress

## Done
* MouseInterface

## Validated
* InterboardCommunication
* Draw_once of Game Control 

## In Progress
* GameControl
* Display

## Pending
* MemoryHandle
* RuleChecking

---

# Implementation Detail

## Interboard Communication
* Interboard reset
    * Implemented in `communication_top.v`
    * This board -> other board
        1. Immediately reset this board's sending and receiving module (`rst` siganl)
        2. Send reset signal to other board by setting all output to 1 (`Ack_out`, `Request_out`, `inter_data_out`)
        3. Wait 10 cycles then reset communication_top (`delayed_rst` signal)
    * Other board -> this board
        1. Immediately reset all the modules upon all input turning to 1 (`Ack_out`, `Request_out`, `inter_data_in`), including communication_top, sending and receiving, and send `interboard_rst` to all the modules
* Sending
* Receiving

## Game Control

### draw_once
* Since the operator `%` is too slow and will cause timing error, I have the write my own modulo function that rely operator on multiple clock cycle

---

# Implementation Note

## Game Control
* Need to get data from `InterboardCommunication` to update state