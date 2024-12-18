`include "game_macro.v"
`include "../message_macro.v"

module handle_advance_state#(
    parameter PLAYER = 0
)(
    input wire clk, 
    input wire rst, 
    input wire interboard_rst,

    input wire start_game,
    input wire [3:0] cur_game_state,
    input wire inter_ready,
    input wire interboard_en,
    input wire [3:0] interboard_msg_type,

    output wire advance_state,

    output wire advance_state_ctrl_en,
    output wire advance_state_ctrl_move_dir,
    output wire [4:0] advance_state_ctrl_block_x,
    output wire [2:0] advance_state_ctrl_block_y,
    output wire [3:0] advance_state_ctrl_msg_type,
    output wire [5:0] advance_state_ctrl_card,
    output wire [2:0] advance_state_ctrl_sel_len
);
endmodule
