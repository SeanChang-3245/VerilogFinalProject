`include "game_macro.v"
`include "../message_macro.v"

// only P1 can advance the game state
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

    output reg advance_state_ctrl_en,
    output wire advance_state_ctrl_move_dir,
    output wire [4:0] advance_state_ctrl_block_x,
    output wire [2:0] advance_state_ctrl_block_y,
    output wire [3:0] advance_state_ctrl_msg_type,
    output wire [5:0] advance_state_ctrl_card,
    output wire [2:0] advance_state_ctrl_sel_len
);

    localparam IDLE = 0;
    localparam WAIT_SEND_ADV_STATE = 1;
    localparam FIN = 2;

    reg [1:0] cur_state, next_state;

    wire player_correct = (PLAYER == `P1 && (cur_game_state == `GAME_INIT || cur_game_state == `GAME_FIN));

    always@(posedge clk) begin
        if(rst || interboard_rst) begin
            cur_state <= IDLE;
        end
        else begin
            cur_state <= next_state;
        end
    end

    assign advance_state = (cur_state == FIN);

    always@(*) begin
        next_state = cur_state;
        if(cur_state == IDLE) begin
            if(interboard_en && interboard_msg_type == `STATE_TURN) begin
                next_state = FIN;
            end 
            else if(player_correct && start_game) begin
                next_state = WAIT_SEND_ADV_STATE;
            end
        end
        else if(cur_state == WAIT_SEND_ADV_STATE && inter_ready) begin
            next_state = FIN;
        end
    end

    always@* begin
        if(cur_state == IDLE && player_correct && start_game) begin
            advance_state_ctrl_en = 1'b1;
        end
        else begin
            advance_state_ctrl_en = 1'b0;
        end
    end


    assign advance_state_ctrl_move_dir = 1'b0;
    assign advance_state_ctrl_block_x = 5'b0;
    assign advance_state_ctrl_block_y = 3'b0;
    assign advance_state_ctrl_msg_type = `STATE_TURN;
    assign advance_state_ctrl_card = 6'b0;
    assign advance_state_ctrl_sel_len = 3'b0;

endmodule
