`include "game_macro.v"
`include "../message_macro.v"

module handle_init_draw#(
    parameter PLAYER = 0
)(
    input wire clk, 
    input wire rst, 
    input wire interboard_rst,

    input wire init_draw_en,
    input wire [3:0] cur_game_state,
    input wire inter_ready,

    input wire [8*18*6-1:0] map,
    input wire [105:0] available_card,

    output wire init_draw_done,

    output reg init_draw_ctrl_en,
    output reg init_draw_ctrl_move_dir,
    output reg [4:0] init_draw_ctrl_block_x,
    output reg [2:0] init_draw_ctrl_block_y,
    output reg [3:0] init_draw_ctrl_msg_type,
    output reg [5:0] init_draw_ctrl_card,
    output reg [2:0] init_draw_ctrl_sel_len
);

    localparam IDLE = 0;
    localparam WAIT_DRAW = 1;
    localparam COUNT = 2;
    localparam WAIT_SEND_TURN = 3;
    localparam FIN = 4;


    wire player_correct = (PLAYER == `P1 && cur_game_state == `GAME_P1_INIT_DRAW) || 
                          (PLAYER == `P2 && cur_game_state == `GAME_P2_INIT_DRAW);

    wire draw_and_place_en;

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

    reg [2:0] cur_state, next_state;
    reg [3:0] counter, counter_next;

    always @(posedge clk) begin
        if(rst || interboard_rst) begin
            cur_state <= IDLE;
            counter <= 0;
        end
        else begin
            cur_state <= next_state;
            counter <= counter_next;
        end
    end

    assign init_draw_done = (cur_state == FIN);
    assign draw_and_place_en = (cur_state == IDLE && player_correct && init_draw_en);

    always@* begin
        next_state = cur_state;
        if(cur_state == IDLE && init_draw_en && player_correct) begin
            next_state = WAIT_DRAW;
        end
        else if(cur_state == WAIT_DRAW && draw_and_place_done) begin
            next_state = COUNT;
        end
        else if(cur_state == COUNT) begin
            if(counter == 14) begin
                next_state =  WAIT_SEND_TURN;
            end
            else begin
                next_state = WAIT_DRAW;
            end
        end
        else if(cur_state == WAIT_SEND_TURN && inter_ready) begin
            next_state = FIN;
        end
        else if(cur_state == FIN) begin
            next_state = IDLE;
        end
    end

    always@* begin
        counter_next = counter;
        if(cur_state == IDLE) begin
            counter_next = 0;
        end
        else if(cur_state == WAIT_DRAW && draw_and_place_done) begin
            counter_next = counter + 1;
        end
    end


    always@* begin
        if(cur_state == WAIT_DRAW || (cur_state == IDLE && player_correct && init_draw_en)) begin
            init_draw_ctrl_en = draw_place_ctrl_en;
        end
        else if(cur_state == COUNT && counter == 14) begin // send STATE_TURN
            init_draw_ctrl_en = 1;
        end
        else begin
            init_draw_ctrl_en = 0;
        end
    end

    always@* begin
        if(cur_state == WAIT_DRAW || (cur_state == IDLE && player_correct && init_draw_en)) begin
            init_draw_ctrl_move_dir = draw_place_ctrl_move_dir;
            init_draw_ctrl_block_x = draw_place_ctrl_block_x;
            init_draw_ctrl_block_y = draw_place_ctrl_block_y;
            init_draw_ctrl_msg_type = draw_place_ctrl_msg_type;
            init_draw_ctrl_card = draw_place_ctrl_card;
            init_draw_ctrl_sel_len = draw_place_ctrl_sel_len;
        end
        else if(cur_state == WAIT_SEND_TURN || (cur_state == COUNT && counter == 14)) begin
            init_draw_ctrl_move_dir = 0;  
            init_draw_ctrl_block_x = 0;
            init_draw_ctrl_block_y = 0;
            init_draw_ctrl_msg_type = `STATE_TURN;
            init_draw_ctrl_card = 0;
            init_draw_ctrl_sel_len = 0;
        end
        else begin
            init_draw_ctrl_move_dir = 0;  
            init_draw_ctrl_block_x = 0;
            init_draw_ctrl_block_y = 0;
            init_draw_ctrl_msg_type = 0;
            init_draw_ctrl_card = 0;
            init_draw_ctrl_sel_len = 0;
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
