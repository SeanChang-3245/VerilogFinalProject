`include "game_macro.v"
`include "../message_macro.v"

module handle_init_draw#(
    parameter PLAYER = 0
)(
    input wire clk, 
    input wire rst, 
    input wire interboard_rst,

    input wire [3:0] cur_game_state,
    input wire inter_ready,

    output wire init_draw_done,

    output wire init_draw_ctrl_en,
    output wire init_draw_ctrl_move_dir,
    output wire [4:0] init_draw_ctrl_block_x,
    output wire [2:0] init_draw_ctrl_block_y,
    output wire [3:0] init_draw_ctrl_msg_type,
    output wire [5:0] init_draw_ctrl_card,
    output wire [2:0] init_draw_ctrl_sel_len
);

    wire player_correct = (PLAYER == `P1 && cur_game_state == `P1_INIT_DRAW) || 
                          (PLAYER == `P2 && cur_game_state == `P2_INIT_DRAW);

endmodule
