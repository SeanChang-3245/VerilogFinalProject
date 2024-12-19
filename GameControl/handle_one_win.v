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

    output reg one_win_ctrl_en,
    output reg one_win_ctrl_move_dir,
    output reg [3:0] one_win_ctrl_msg_type,
    output reg [5:0] one_win_ctrl_card,
    output reg [2:0] one_win_ctrl_sel_len,
    output reg [4:0] one_win_ctrl_block_x,
    output reg [2:0] one_win_ctrl_block_y
);

    localparam IDLE = 0;
    localparam WAIT_SEND_WIN = 1;
    localparam FIN = 2;

    wire player_correct = (PLAYER == `P1 && cur_game_state == `GAME_P1_WAIT_IN) || 
                          (PLAYER == `P2 && cur_game_state == `GAME_P2_WAIT_IN);

    reg [1:0] cur_state, next_state;

    always@(posedge clk) begin
        if(rst || interboard_rst) begin
            cur_state <= IDLE;
        end
        else begin
            cur_state <= next_state;
        end
    end

    assign one_win = (cur_state == FIN);

    always@(*) begin
        next_state = cur_state;
        if(cur_state == IDLE) begin
            if(interboard_en && interboard_msg_type == `STATE_I_WIN) begin
                next_state = FIN;
            end 
            else if(player_correct && done_and_next && my_card_cnt == 0) begin
                next_state = WAIT_SEND_WIN;
            end
        end
        else if(cur_state == WAIT_SEND_WIN && inter_ready) begin
            next_state = FIN;
        end
        else if(cur_state == WAIT_SEND_WIN) begin
            next_state = IDLE;
        end
    end

    always @(*) begin
        if(cur_state == IDLE && player_correct && done_and_next && my_card_cnt == 0) begin
            one_win_ctrl_en = 1;
        end
        else begin
            one_win_ctrl_en = 0;
        end
    end

    always@* begin
        one_win_ctrl_msg_type = `STATE_I_WIN;
        one_win_ctrl_move_dir = 0;
        one_win_ctrl_card = 0;
        one_win_ctrl_sel_len = 0;
        one_win_ctrl_block_x = 0;
        one_win_ctrl_block_y = 0;
    end    


endmodule