`include "game_macro.v"
`include "../message_macro.v"

module handle_switch_turn#(
    parameter PLAYER = 0
)(
    input wire clk, 
    input wire rst, 
    input wire interboard_rst,

    input wire done_and_next,
    input wire draw_and_next,
    input wire can_done,
    input wire can_draw,
    input wire [3:0] cur_game_state,
    input wire inter_ready,
    
    input wire interboard_en,
    input wire [3:0] interboard_msg_type,

    output wire switch_turn,

    output wire switch_turn_ctrl_en,
    output wire switch_turn_ctrl_move_dir,
    output wire [4:0] switch_turn_ctrl_block_x,
    output wire [2:0] switch_turn_ctrl_block_y,
    output wire [3:0] switch_turn_ctrl_msg_type,
    output wire [5:0] switch_turn_ctrl_card,
    output wire [2:0] switch_turn_ctrl_sel_len
);

endmodule