`include "game_macro.v"
`include "../message_macro.v"

module handle_one_win#(
    parameter PLAYER = 0
)(
    input wire clk, 
    input wire rst, 
    input wire interboard_rst,

    input wire done_and_next,
    input wire [6:0] my_card_cnt,
    input wire [3:0] cur_game_state,
    input wire inter_ready,

    input wire interboard_en,
    input wire [3:0] interboard_msg_type,

    output wire one_win,

    output wire one_win_ctrl_en,
    output wire one_win_ctrl_move_dir,
    output wire [3:0] one_win_ctrl_msg_type,
    output wire [5:0] one_win_ctrl_card,
    output wire [2:0] one_win_ctrl_sel_len,
    output wire [4:0] one_win_ctrl_block_x,
    output wire [2:0] one_win_ctrl_block_y
);
endmodule