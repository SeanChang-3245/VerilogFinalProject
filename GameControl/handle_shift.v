`include "game_macro.v"
`include "../message_macro.v"

module handle_shift#(
    parameter PLAYER = 0
)(
    input wire clk, 
    input wire rst, 
    input wire interboard_rst,

    input wire move_left,
    input wire move_right,
    input wire shift_en,
    input wire [3:0] cur_game_state,
    input wire inter_ready,

    output wire shift_done,

    output wire shift_ctrl_en,
    output wire shift_ctrl_move_dir,
    output wire [4:0] shift_ctrl_block_x,
    output wire [2:0] shift_ctrl_block_y,
    output wire [3:0] shift_ctrl_msg_type,
    output wire [5:0] shift_ctrl_card,
    output wire [2:0] shift_ctrl_sel_len
);


endmodule