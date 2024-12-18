`include "game_macro.v"
`include "../message_macro.v"

module handle_move#(
    parameter PLAYER = 0
)(
    input wire clk, 
    input wire rst, 
    input wire interboard_rst,

    input wire move_en,
    input wire [3:0] cur_game_state,
    input wire inter_ready,
    input wire valid_card_take,
    input wire valid_card_down,

    input wire l_click,
    input wire mouse_inblock,
    input wire [4:0] mouse_block_x,
    input wire [2:0] mouse_block_y,

    output wire move_done,

    output wire move_ctrl_en,
    output wire move_ctrl_move_dir,
    output wire [4:0] move_ctrl_block_x,
    output wire [2:0] move_ctrl_block_y,
    output wire [3:0] move_ctrl_msg_type,
    output wire [5:0] move_ctrl_card,
    output wire [2:0] move_ctrl_sel_len
);

endmodule