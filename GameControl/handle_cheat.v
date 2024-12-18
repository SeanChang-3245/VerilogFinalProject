`include "../message_macro.v"
`include "game_macro.v"

module handle_cheat#(
    parameter PLAYER = 0
)(
    input wire clk, 
    input wire rst, 
    input wire interboard_rst,

    input wire five_r_click,
    input wire [3:0] cur_game_state,
    input wire inter_ready,

    input wire interboard_en,
    input wire [3:0] interboard_msg_type,

    output wire cheat_activate,
    
    output wire cheat_ctrl_en,
    output wire cheat_ctrl_move_dir,
    output wire [4:0] cheat_ctrl_block_x,
    output wire [2:0] cheat_ctrl_block_y,
    output wire [3:0] cheat_ctrl_msg_type,
    output wire [5:0] cheat_ctrl_card,
    output wire [2:0] cheat_ctrl_sel_len
);
endmodule