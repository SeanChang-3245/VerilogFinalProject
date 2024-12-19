`include "game_macro.v"
`include "../message_macro.v"

module handle_move#(
    parameter PLAYER = 0
)(
    input wire clk, 
    input wire rst, 
    input wire interboard_rst,

    // input wire move_en,
    input wire [3:0] cur_game_state,
    input wire inter_ready,
    input wire valid_card_take,
    input wire valid_card_down,
    input wire [8*18*6-1:0] map,

    input wire l_click,
    input wire mouse_inblock,           // may be not neccessary if valid_card_take and valid_card_down are used
    input wire [4:0] mouse_block_x,
    input wire [2:0] mouse_block_y,

    output wire move_done,
    output reg [8*18-1:0] move_sel_card,

    output reg move_ctrl_en,
    output reg move_ctrl_move_dir,
    output reg [4:0] move_ctrl_block_x,
    output reg [2:0] move_ctrl_block_y,
    output reg [3:0] move_ctrl_msg_type,
    output reg [5:0] move_ctrl_card,
    output reg [2:0] move_ctrl_sel_len
);

    localparam IDLE = 0;
    localparam WAIT_SEND_TAKE = 1;
    localparam WAIT_PLAYER_CLICK = 2;
    localparam WAIT_SEND_DOWN = 3;
    localparam FIN = 4;

    wire player_correct = (PLAYER == `P1 && cur_game_state == `GAME_P1_WAIT_IN) ||
                          (PLAYER == `P2 && cur_game_state == `GAME_P2_WAIT_IN);

    reg [2:0] cur_state, next_state;
    // reg [4:0] click_mouse_block_x, click_mouse_block_x_next;
    // reg [2:0] click_mouse_block_y, click_mouse_block_y_next;
    reg [6:0] taken_card, taken_card_next;
    reg take_msg_type, down_msg_type;
    reg [8*18-1:0] move_sel_card_next;

    always@(posedge clk) begin
        if(rst || interboard_rst) begin
            cur_state <= IDLE;
            taken_card <= 54;
            move_sel_card <= 0;
        end
        else begin
            cur_state <= next_state;
            taken_card <= taken_card_next;
            move_sel_card <= move_sel_card_next;
        end
    end

    assign move_done = (cur_state == FIN);

    always@* begin
        if(mouse_block_y < 6) begin
            take_msg_type = `TABLE_TAKE;
            down_msg_type = `TABLE_DOWN;
        end
        else begin
            take_msg_type = `HAND_DOWN;
            down_msg_type = `HAND_DOWN;
        end
    end

    always@* begin
        move_sel_card_next = move_sel_card;
        if(cur_state == WAIT_PLAYER_CLICK && valid_card_down && l_click && player_correct) begin
            move_sel_card_next = mouse_block_x + 18*mouse_block_y;
        end
        else if(cur_state == FIN) begin
            move_sel_card_next = 0;
        end
    end

    always@* begin
        taken_card_next = taken_card;
        if(cur_state == IDLE && valid_card_take && l_click && player_correct) begin
            taken_card_next = map[(18*mouse_block_y + mouse_block_x)*6 +: 6];
        end
    end

    always@* begin
        next_state = cur_state;
        if(cur_state == IDLE && valid_card_take && l_click && player_correct) begin
            next_state = WAIT_SEND_TAKE;
        end
        else if(cur_state == WAIT_SEND_TAKE && inter_ready) begin
            next_state = WAIT_PLAYER_CLICK;
        end
        else if(cur_state == WAIT_PLAYER_CLICK && valid_card_down && l_click && player_correct) begin
            next_state = WAIT_SEND_DOWN;
        end
        else if(cur_state == WAIT_SEND_DOWN && inter_ready) begin
            next_state = FIN;
        end
        else if(cur_state == FIN) begin
            next_state = IDLE;
        end
    end

    always@* begin
        if(cur_state == IDLE && player_correct && valid_card_take && l_click) begin
            move_ctrl_en = 1;
        end
        else if(cur_state == WAIT_PLAYER_CLICK && player_correct && valid_card_down && l_click) begin
            move_ctrl_en = 1;
        end
        else begin
            move_ctrl_en = 0;
        end
    end

    always @(*) begin
        if (cur_state == IDLE || cur_state == WAIT_SEND_TAKE) begin
            move_ctrl_move_dir = 0;
            move_ctrl_block_x = mouse_block_x;
            move_ctrl_block_y = mouse_block_y;
            move_ctrl_msg_type = take_msg_type;
            move_ctrl_card = 0;
            move_ctrl_sel_len = 0;
        end
        else if (cur_state == WAIT_PLAYER_CLICK || cur_state == WAIT_SEND_DOWN) begin
            move_ctrl_move_dir = 0;
            move_ctrl_block_x = mouse_block_x;
            move_ctrl_block_y = mouse_block_y;
            move_ctrl_msg_type = down_msg_type;
            move_ctrl_card = taken_card;
            move_ctrl_sel_len = 0;
        end
        else begin
            move_ctrl_move_dir = 0;
            move_ctrl_block_x = 0;
            move_ctrl_block_y = 0;
            move_ctrl_msg_type = 0;
            move_ctrl_card = 0;
            move_ctrl_sel_len = 0;
        end
    end

endmodule