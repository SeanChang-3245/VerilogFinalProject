module draw_one_place_send_msg(
    input wire clk,
    input wire rst, 
    input wire interboard_rst,
    input wire en,
    input wire inter_ready,                  // from interboard, mean it can transmit another action
    input wire [8*18*6-1:0] map,
    input wire [105:0] available_card,

    output wire draw_ctrl_en,
    output wire draw_ctrl_move_dir,
    output wire [4:0] draw_ctrl_block_x,
    output wire [2:0] draw_ctrl_block_y,
    output wire [3:0] draw_ctrl_msg_type,
    output wire [5:0] draw_ctrl_card,       // need to convert card index to card type 0~105 -> 0~53
    output wire [2:0] draw_ctrl_sel_len
);

    localparam IDLE = 0;
    localparam WAIT_DRAW = 1;
    localparam WAIT_INTER_SEND_DRAW = 2;
    localparam WAIT_PLACE_CARD = 3;
    localparam WAIT_INTER_SEND_PLACE = 4;

    wire draw_done, draw_ready;
    

    reg draw_one, draw_one_next;
    reg [2:0] cur_state, next_state;

    always@(posedge clk) begin
        if(rst || interboard_rst) begin
            cur_state <= IDLE;        
            draw_one <= 0;
        end
        else begin
            cur_state <= next_state;
            draw_one <= draw_one_next;
        end
    end

    always@(*) begin
        next_state = cur_state;
        if(cur_state == IDLE && en) begin
            next_state = WAIT_DRAW;
        end
        else if(cur_state == WAIT_DRAW && draw_done) begin // need to activate send before enter wait inter send
            next_state = WAIT_INTER_SEND_DRAW;
        end
        else if(cur_state == WAIT_INTER_SEND_DRAW && inter_ready) begin  
            next_state = WAIT_PLACE_CARD;
        end
        else if(cur_state == WAIT_PLACE_CARD) begin
            next_state = WAIT_INTER_SEND_PLACE;
        end
        else if(cur_state == WAIT_INTER_SEND_PLACE && inter_ready) begin
            next_state = IDLE;
        end
    end

    always @(*) begin
        draw_one_next = 0;
        if(cur_state == IDLE && en) begin
            draw_one_next = 1;
        end
    end

endmodule