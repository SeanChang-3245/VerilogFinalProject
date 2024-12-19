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
    input wire [8*18*6-1:0] map,
    input wire [105:0] available_card,
    input wire [6:0] my_card_cnt,
    
    input wire interboard_en,
    input wire [3:0] interboard_msg_type,

    output wire switch_turn,

    output reg switch_turn_ctrl_en,
    output reg switch_turn_ctrl_move_dir,
    output reg [4:0] switch_turn_ctrl_block_x,
    output reg [2:0] switch_turn_ctrl_block_y,
    output reg [3:0] switch_turn_ctrl_msg_type,
    output reg [5:0] switch_turn_ctrl_card,
    output reg [2:0] switch_turn_ctrl_sel_len
);

    localparam IDLE = 0;
    localparam WAIT_DRAW = 1;
    localparam WAIT_SEND_TURN = 2;
    localparam FIN = 3;

    // output from draw_one_place_send_msg
    wire draw_and_place_done;
    wire draw_and_place_ready;
    wire draw_place_ctrl_en;
    wire draw_place_ctrl_move_dir;
    wire [4:0] draw_place_ctrl_block_x;
    wire [2:0] draw_place_ctrl_block_y;
    wire [3:0] draw_place_ctrl_msg_type;
    wire [5:0] draw_place_ctrl_card;
    wire [2:0] draw_place_ctrl_sel_len;

    reg [1:0] cur_state, next_state;
    reg draw_and_place_en;
    wire player_correct = (PLAYER == `P1 && cur_game_state == `GAME_P1_WAIT_IN) || 
                          (PLAYER == `P2 && cur_game_state == `GAME_P2_WAIT_IN);

    always @(posedge clk) begin
        if(rst || interboard_rst) begin
            cur_state <= IDLE;
        end
        else begin
            cur_state <= next_state;
        end
    end

    assign switch_turn = (cur_state == FIN);
    
    always@* begin
        next_state = cur_state;
        if(cur_state == IDLE) begin
            if(player_correct && draw_and_next && can_draw) begin
                next_state = WAIT_DRAW;
            end
            else if(player_correct && done_and_next && can_done && my_card_cnt != 0) begin
                next_state = WAIT_SEND_TURN;
            end
            else if(interboard_en && interboard_msg_type == `STATE_TURN) begin
                next_state = FIN;
            end
        end
        else if(cur_state == WAIT_DRAW && draw_and_place_done) begin
            next_state = WAIT_SEND_TURN;
        end
        else if(cur_state == WAIT_SEND_TURN && inter_ready) begin
            next_state = FIN;
        end
        else if(cur_state == FIN) begin
            next_state = IDLE;
        end
    end

    always@* begin
        if(cur_state == IDLE && player_correct && draw_and_next && can_draw) begin
            draw_and_place_en = 1;
        end
        else begin
            draw_and_place_en = 0;
        end
    end

    always@* begin
        if(cur_state == IDLE && player_correct && draw_and_next && can_draw || cur_state == WAIT_DRAW) begin
            switch_turn_ctrl_en = draw_place_ctrl_en;
        end
        else if(cur_state == WAIT_DRAW && draw_and_place_done) begin
            switch_turn_ctrl_en = 1;
        end
        else if(cur_state == IDLE && player_correct && done_and_next && can_done && my_card_cnt != 0) begin
            switch_turn_ctrl_en = 1;
        end
        else begin
            switch_turn_ctrl_en = 0;
        end
    end

    always@* begin
        if(cur_state == IDLE && player_correct && draw_and_next && can_draw || cur_state == WAIT_DRAW) begin
            switch_turn_ctrl_move_dir = draw_place_ctrl_move_dir;
            switch_turn_ctrl_block_x = draw_place_ctrl_block_x;
            switch_turn_ctrl_block_y = draw_place_ctrl_block_y;
            switch_turn_ctrl_msg_type = draw_place_ctrl_msg_type;
            switch_turn_ctrl_card = draw_place_ctrl_card;
            switch_turn_ctrl_sel_len = draw_place_ctrl_sel_len;
        end 
        else if(cur_state == IDLE && player_correct && done_and_next && can_done && my_card_cnt != 0 || cur_state == WAIT_SEND_TURN) begin
            switch_turn_ctrl_move_dir = 0;
            switch_turn_ctrl_block_x = 0;
            switch_turn_ctrl_block_y = 0;
            switch_turn_ctrl_msg_type = `STATE_TURN;
            switch_turn_ctrl_card = 0;
            switch_turn_ctrl_sel_len = 0;
        end
        else if(cur_state == WAIT_DRAW && draw_and_place_done) begin
            switch_turn_ctrl_move_dir = 0;
            switch_turn_ctrl_block_x = 0;
            switch_turn_ctrl_block_y = 0;
            switch_turn_ctrl_msg_type = `STATE_TURN;
            switch_turn_ctrl_card = 0;
            switch_turn_ctrl_sel_len = 0;
        end
        else begin
            switch_turn_ctrl_move_dir = 0;
            switch_turn_ctrl_block_x = 0;
            switch_turn_ctrl_block_y = 0;
            switch_turn_ctrl_msg_type = 0;
            switch_turn_ctrl_card = 0;
            switch_turn_ctrl_sel_len = 0;
        end
    end

    draw_one_place_send_msg draw_place_inst(
        .clk(clk),
        .rst(rst),
        .interboard_rst(interboard_rst),

        .draw_and_place_en(draw_and_place_en),
        .inter_ready(inter_ready),
        .map(map),
        .available_card(available_card),

        .draw_and_place_ready(draw_and_place_ready),
        .draw_and_place_done(draw_and_place_done),

        .draw_place_ctrl_en(draw_place_ctrl_en),
        .draw_place_ctrl_move_dir(draw_place_ctrl_move_dir),
        .draw_place_ctrl_block_x(draw_place_ctrl_block_x),
        .draw_place_ctrl_block_y(draw_place_ctrl_block_y),
        .draw_place_ctrl_msg_type(draw_place_ctrl_msg_type),
        .draw_place_ctrl_card(draw_place_ctrl_card),
        .draw_place_ctrl_sel_len(draw_place_ctrl_sel_len)
    );

endmodule