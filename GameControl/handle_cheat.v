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
    
    output reg cheat_ctrl_en,
    output reg cheat_ctrl_move_dir,
    output reg [4:0] cheat_ctrl_block_x,
    output reg [2:0] cheat_ctrl_block_y,
    output reg [3:0] cheat_ctrl_msg_type,
    output reg [5:0] cheat_ctrl_card,
    output reg [2:0] cheat_ctrl_sel_len
);

    localparam IDLE = 0;
    localparam WAIT_SEND_CHEAT = 1;
    localparam FIN = 2;

    reg [1:0] cur_state, next_state;

    wire player_correct = (PLAYER == `P1 && (cur_game_state == `GAME_P1_WAIT_IN || cur_game_state == `GAME_P1_MOVE || cur_game_state == `GAME_P1_SHIFT)) || 
                          (PLAYER == `P2 && (cur_game_state == `GAME_P2_WAIT_IN || cur_game_state == `GAME_P2_MOVE || cur_game_state == `GAME_P2_SHIFT));
    
    always@(posedge clk) begin
        if(rst || interboard_rst) begin
            cur_state <= IDLE;
        end
        else begin
            cur_state <= next_state;
        end
    end

    assign cheat_activate = (cur_state == FIN);

    always@* begin
        next_state = cur_state;
        if(cur_state == IDLE) begin
            if(interboard_en && interboard_msg_type == `STATE_CHEAT) begin
                next_state = FIN;
            end
            else if(player_correct && five_r_click) begin
                next_state = WAIT_SEND_CHEAT;
            end
        end
        else if(cur_state == WAIT_SEND_CHEAT && inter_ready) begin
            next_state = FIN;
        end
    end

    always@* begin
        if(cur_state == IDLE && player_correct && five_r_click) begin
            cheat_ctrl_en = 1;
        end
        else begin
            cheat_ctrl_en = 0;
        end
    end

    always@* begin
        cheat_ctrl_msg_type = `STATE_CHEAT;
        cheat_ctrl_move_dir = 0;
        cheat_ctrl_block_x = 0;
        cheat_ctrl_block_y = 0;
        cheat_ctrl_card = 0;
        cheat_ctrl_sel_len = 0;
    end

endmodule