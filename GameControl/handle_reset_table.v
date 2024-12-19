`include "game_macro.v"
`include "../message_macro.v"

module handle_reset_table#(
    parameter PLAYER = 0
)(
    input wire clk, 
    input wire rst, 
    input wire interboard_rst,

    input wire reset_table_raw,
    input wire [3:0] cur_game_state,
    input wire inter_ready,

    input wire interboard_en,
    input wire [3:0] interboard_msg_type,

    output wire reset_table,

    output reg reset_table_ctrl_en,
    output reg reset_table_ctrl_move_dir,
    output reg [4:0] reset_table_ctrl_block_x,
    output reg [2:0] reset_table_ctrl_block_y,
    output reg [3:0] reset_table_ctrl_msg_type,
    output reg [5:0] reset_table_ctrl_card,
    output reg [2:0] reset_table_ctrl_sel_len
);

    localparam IDLE = 0;
    localparam WAIT_SEND_RST = 1;
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

    assign reset_table = (cur_state == FIN);

    always@* begin
        next_state = cur_state;
        if(cur_state == IDLE) begin
            if(interboard_en && interboard_msg_type == `STATE_RST_TABLE) begin
                next_state = FIN;
            end
            else if(player_correct && reset_table_raw) begin
                next_state = WAIT_SEND_RST;
            end
        end
        else if(cur_state == WAIT_SEND_RST && inter_ready) begin
            next_state = FIN;
        end
        else if(cur_state == FIN) begin
            next_state = IDLE;
        end
    end

    always @(*) begin
        if(cur_state == IDLE && player_correct && reset_table_raw) begin
            reset_table_ctrl_en = 1;
        end
        else begin
            reset_table_ctrl_en = 0;
        end
    end

    always@* begin
        reset_table_ctrl_msg_type = `STATE_RST_TABLE;
        reset_table_ctrl_block_x = 0;
        reset_table_ctrl_block_y = 0;
        reset_table_ctrl_card = 0;
        reset_table_ctrl_sel_len = 0;
        reset_table_ctrl_move_dir = 0;
    end

endmodule

