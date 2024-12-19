`include "../message_macro.v"

module draw_one_place_send_msg(
    input wire clk,
    input wire rst, 
    input wire interboard_rst,
    
    input wire draw_and_place_en,                           // from upper layer, mean it can start a new action
    input wire inter_ready,                  // from interboard, mean it can transmit another action
    input wire [8*18*6-1:0] map,
    input wire [105:0] available_card,

    output wire draw_and_place_ready,        // to upper layer, indicating this moldule is ready to draw and place
    output wire draw_and_place_done,         // to upper layer, indicating this module has finished draw and place (one-pulse)

    output reg draw_place_ctrl_en,
    output reg draw_place_ctrl_move_dir,
    output reg [4:0] draw_place_ctrl_block_x,
    output reg [2:0] draw_place_ctrl_block_y,
    output reg [3:0] draw_place_ctrl_msg_type,
    output reg [5:0] draw_place_ctrl_card,       // need to convert card index to card type 0~105 -> 0~53
    output reg [2:0] draw_place_ctrl_sel_len
);

    localparam IDLE = 0;
    localparam WAIT_DRAW = 1;
    localparam WAIT_INTER_SEND_DRAW = 2;
    localparam WAIT_PLACE_CARD = 3;
    localparam WAIT_INTER_SEND_PLACE = 4;
    localparam FIN = 5;

    // draw_once output 
    wire draw_one_done, draw_one_ready;
    wire [6:0] drawn_card_idx;
    wire [5:0] card_place;

    reg [5:0] drawn_card_type;                  // convert card_index (from available card) to card_type 
    reg [2:0] cur_state, next_state;
    wire draw_one_en = (cur_state == IDLE && draw_and_place_en);


    draw_once draw_once_inst (
        .clk(clk),
        .rst(rst), 
        .interboard_rst(interboard_rst),
        
        .draw_one(draw_one_en),
        .available_card(available_card),

        .done(draw_one_done),
        .ready(draw_one_ready),
        .drawn_card_idx(drawn_card_idx)
    );
    
    find_draw_card_place find_draw_card_place_inst (
        .clk(clk),
        .rst(rst), 
        .interboard_rst(interboard_rst),
        .map(map),
        .card_place(card_place)
    );

    always@(posedge clk) begin
        if(rst || interboard_rst) begin
            cur_state <= IDLE;
        end
        else begin
            cur_state <= next_state;
        end
    end

    assign draw_and_place_done = (cur_state == FIN);
    assign draw_and_place_ready = (cur_state == IDLE);

    always@(*) begin
        next_state = cur_state;
        if(cur_state == IDLE && draw_and_place_en) begin
            next_state = WAIT_DRAW;
        end
        else if(cur_state == WAIT_DRAW && draw_one_done) begin // need to activate send before enter wait inter send
            next_state = WAIT_INTER_SEND_DRAW;
        end
        else if(cur_state == WAIT_INTER_SEND_DRAW && inter_ready) begin  
            next_state = WAIT_PLACE_CARD;
        end
        else if(cur_state == WAIT_PLACE_CARD) begin
            next_state = WAIT_INTER_SEND_PLACE;
        end
        else if(cur_state == WAIT_INTER_SEND_PLACE && inter_ready) begin
            next_state = FIN;
        end
        else if(cur_state == FIN) begin
            next_state = IDLE;
        end
    end

    always@* begin
        if(cur_state == WAIT_DRAW && draw_one_done) begin
            draw_place_ctrl_en = 1;
        end   
        else if (cur_state == WAIT_PLACE_CARD) begin
            draw_place_ctrl_en = 1;
        end     
        else begin
            draw_place_ctrl_en = 0;
        end
    end

    always@* begin
        if(cur_state == WAIT_DRAW || cur_state == WAIT_INTER_SEND_DRAW) begin
            draw_place_ctrl_move_dir = 0;
            draw_place_ctrl_sel_len = 0;
            draw_place_ctrl_card = drawn_card_type; 
            draw_place_ctrl_block_x = 0;
            draw_place_ctrl_block_y = 0;
            draw_place_ctrl_msg_type = `DECK_DRAW;
        end
        else if(cur_state == WAIT_PLACE_CARD || cur_state == WAIT_INTER_SEND_PLACE) begin
            draw_place_ctrl_move_dir = 0;
            draw_place_ctrl_sel_len = 0;
            draw_place_ctrl_card = drawn_card_type; 
            draw_place_ctrl_block_x = card_place >= 18 ? (card_place-18) : card_place;
            draw_place_ctrl_block_y = card_place >= 18 ? 7 : 6;
            draw_place_ctrl_msg_type = `HAND_DOWN;
        end
        else begin
            draw_place_ctrl_move_dir = 0;
            draw_place_ctrl_sel_len = 0;
            draw_place_ctrl_card = 0; 
            draw_place_ctrl_block_x = 0;
            draw_place_ctrl_block_y = 0;
            draw_place_ctrl_msg_type = 15;
        end
    end

    always@* begin
        if(drawn_card_idx == 52 || drawn_card_idx == 53) begin
            drawn_card_type = drawn_card_idx;
        end
        else if(drawn_card_idx >= 54) begin
            drawn_card_type = drawn_card_idx - 54;
        end 
        else begin
            drawn_card_type = drawn_card_idx;
        end
    end

endmodule