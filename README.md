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

## Display

### Display_card
* Store `map` as 6-bit array in the module, making it simpe to use position to fetch the corrosponding message type
* Use sel_card, current pixel_x, and pixel_y to determine whether the selected frame need to be shown or not.
* Use h_cnt to calculate position x. range from 0 to 18
* Use v_cnt to calculate position y. range from o to 8

### Mem_pixel
* Store 4 sets of number cards in 4 differnt memories, and 2 joke cards in 2 memories respectively.
* Use `card_type`, `pixel_x`, and `pixel_y` to determine which memory and what the pixel_addr
* When the card_type is `none (label as 54)`, display the color same as `background (68A)`.
---

# Implementation Note

## Game Control
* Need to get data from `InterboardCommunication` to update state
* Use matrix to check whether a card can be put back to hand
* Need to check whether one card can be put down (the block must be empty)