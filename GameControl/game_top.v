`include "game_macro.v"
`include "../message_macro.v"

module GameControl_top #(
    parameter PLAYER = 0
)(
    input wire clk,
    input wire rst, 
    input wire interboard_rst,
    
    // from user input
    input wire shift_en,                   // from Player_top, decide which operation the player is doing, shift or take/down, not one-pulsef
    input wire start_game,
    input wire rule_valid,
    input wire move_left,
    input wire move_right,
    input wire reset_table_raw,
    input wire done_and_next,
    input wire draw_and_next,

    // from memory
    input wire [6:0] oppo_card_cnt,
    input wire [105:0] available_card,    // used to determine which cards can be drawn
    input wire [8*18*6-1:0] map,

    // from InterboardCommunication
    input wire inter_ready,                // from InterboardCommunication, indicate the module is ready to transmit action done to other side
    input wire interboard_en,             // from InterboardCommunication, one-pulse signal to indicate the transmitted data is valid
    input wire [3:0] interboard_msg_type, // from interboard, used to advance state in FSM
    // input wire [5:0] picked_card,         // from memory, indicate the card that is clicked

    // from MouseInterface
    input wire l_click,
    input wire five_r_click,
    input wire mouse_inblock,
    input wire [9:0] mouse_x,
    input wire [8:0] mouse_y,
    input wire [4:0] mouse_block_x,       // mouse information
    input wire [2:0] mouse_block_y,       // mouse information

    output wire can_done,                 // indicate the player can finish his turn and switch turn
    output wire can_draw,                 // indicate the player can draw a card and switch turn
    output reg transmit,                 // to InterboardCommunication and Memory, indicate it's this player's turn to transmit data (move)
    output reg ctrl_en,
    output reg ctrl_move_dir,
    output reg [4:0] ctrl_block_x,       // protocol information 
    output reg [2:0] ctrl_block_y,       // protocol information
    output reg [3:0] ctrl_msg_type,
    output reg [5:0] ctrl_card,
    output reg [2:0] ctrl_sel_len,

    output reg [8*18-1:0] sel_card       // to Display, indicate which cards are selected
);   

    // handle advance state output
    wire advance_state;
    wire advance_state_ctrl_en;
    wire advance_state_ctrl_move_dir;
    wire [4:0] advance_state_ctrl_block_x;
    wire [2:0] advance_state_ctrl_block_y;
    wire [3:0] advance_state_ctrl_msg_type;
    wire [5:0] advance_state_ctrl_card;
    wire [2:0] advance_state_ctrl_sel_len;

    // handle cheat output
    wire cheat_activate;
    wire cheat_ctrl_en;
    wire cheat_ctrl_move_dir;
    wire [4:0] cheat_ctrl_block_x;
    wire [2:0] cheat_ctrl_block_y;
    wire [3:0] cheat_ctrl_msg_type;
    wire [5:0] cheat_ctrl_card;
    wire [2:0] cheat_ctrl_sel_len;

    // handle init draw output
    wire init_draw_done;
    wire init_draw_ctrl_en;
    wire init_draw_ctrl_move_dir;
    wire [4:0] init_draw_ctrl_block_x;
    wire [2:0] init_draw_ctrl_block_y;
    wire [3:0] init_draw_ctrl_msg_type;
    wire [5:0] init_draw_ctrl_card;
    wire [2:0] init_draw_ctrl_sel_len;

    // handle move output
    wire move_done;
    wire [8*18-1:0] move_sel_card;
    wire move_ctrl_en;
    wire move_ctrl_move_dir;
    wire [4:0] move_ctrl_block_x;
    wire [2:0] move_ctrl_block_y;
    wire [3:0] move_ctrl_msg_type;
    wire [5:0] move_ctrl_card;
    wire [2:0] move_ctrl_sel_len;

    // handle one win output
    wire one_win;
    wire one_win_ctrl_en;
    wire one_win_ctrl_move_dir;
    wire [3:0] one_win_ctrl_msg_type;
    wire [5:0] one_win_ctrl_card;
    wire [2:0] one_win_ctrl_sel_len;
    wire [4:0] one_win_ctrl_block_x;
    wire [2:0] one_win_ctrl_block_y;

    // handle reset table output
    wire reset_table;
    wire reset_table_ctrl_en;
    wire reset_table_ctrl_move_dir;
    wire [4:0] reset_table_ctrl_block_x;
    wire [2:0] reset_table_ctrl_block_y;
    wire [3:0] reset_table_ctrl_msg_type;
    wire [5:0] reset_table_ctrl_card;
    wire [2:0] reset_table_ctrl_sel_len;

    // handle shift output
    wire shift_done;
    wire [8*18-1:0] shift_sel_card;
    wire shift_ctrl_en;
    wire shift_ctrl_move_dir;
    wire [4:0] shift_ctrl_block_x;
    wire [2:0] shift_ctrl_block_y;
    wire [3:0] shift_ctrl_msg_type;
    wire [5:0] shift_ctrl_card;
    wire [2:0] shift_ctrl_sel_len;

    // handle switch turn output
    wire switch_turn;
    wire switch_turn_ctrl_en;
    wire switch_turn_ctrl_move_dir;
    wire [4:0] switch_turn_ctrl_block_x;
    wire [2:0] switch_turn_ctrl_block_y;
    wire [3:0] switch_turn_ctrl_msg_type;
    wire [5:0] switch_turn_ctrl_card;
    wire [2:0] switch_turn_ctrl_sel_len;

    // Define local variable
    reg [3:0] cur_game_state, next_game_state;
    reg valid_card_take, valid_card_down;
    reg [6:0] my_card_cnt;
    reg my_turn;
    reg resetting_table, resetting_table_next;  // when reset_table is called from this board, 
                                                // interboard transmission should be controlled by handle_reset_table
    reg cheatting, cheatting_next;              // when five_r_click is called from this board, 
                                                // interboard transmission should be controlled by handle_cheat
    reg init_draw_en;

    always @(posedge clk) begin
        if(rst || interboard_rst) begin
            cur_game_state <= `GAME_INIT;
            resetting_table <= 0;
            cheatting <= 0;
        end
        else begin
            cur_game_state <= next_game_state;
            resetting_table <= resetting_table_next;
            cheatting <= cheatting_next;
        end
    end

    always@* begin
        if(cur_game_state == `GAME_INIT && advance_state) begin
            init_draw_en = 1;
        end
        else if (cur_game_state == `GAME_P1_INIT_DRAW && (init_draw_done || interboard_en && interboard_msg_type == `STATE_TURN))begin
            init_draw_en = 1;
        end
        else begin
            init_draw_en = 0;
        end
    end

    always@* begin
        if(PLAYER == `P1) begin
            if(cur_game_state == `GAME_FIN || cur_game_state == `GAME_INIT || cur_game_state == `GAME_P1_INIT_DRAW ||
               cur_game_state == `GAME_P1_WAIT_IN || cur_game_state == `GAME_P1_MOVE || cur_game_state == `GAME_P1_SHIFT) begin
                transmit = 1;
            end
            else begin
                transmit = 0;
            end
        end
        else begin
            if(cur_game_state == `GAME_P2_INIT_DRAW || cur_game_state == `GAME_P2_WAIT_IN || 
               cur_game_state == `GAME_P2_MOVE || cur_game_state == `GAME_P2_SHIFT) begin
                transmit = 1;
            end
            else begin
                transmit = 0;
            end
        end
    end

    always@* begin
        sel_card = 0;
        if(cur_game_state == `GAME_P1_WAIT_IN || cur_game_state == `GAME_P1_MOVE || 
           cur_game_state == `GAME_P2_WAIT_IN || cur_game_state == `GAME_P2_MOVE) begin
            sel_card = move_sel_card;
        end
        else if(cur_game_state == `GAME_P1_SHIFT || cur_game_state == `GAME_P2_SHIFT) begin
            sel_card = shift_sel_card;
        end
    end

    always@* begin
        resetting_table_next = resetting_table;
        if(reset_table_raw && `GAME_P1_WAIT_IN <= cur_game_state && cur_game_state <= `GAME_P2_SHIFT) begin
            resetting_table_next = 1;
        end
        else if(reset_table) begin // reset table signal had send globally
            resetting_table_next = 0;
        end
    end

    always@* begin
        cheatting_next = cheatting;
        if(five_r_click && `GAME_P1_WAIT_IN <= cur_game_state && cur_game_state <= `GAME_P2_SHIFT) begin
            cheatting_next = 1;
        end
        else if(cheat_activate) begin // cheat activate signal had send globally
            cheatting_next = 0;
        end
    end

    always@* begin
        valid_card_take = 1;
    end

    always@* begin
        valid_card_down = 1;
    end

    always@* begin
        my_card_cnt = 1;
    end

    always@* begin
        next_game_state = cur_game_state;
        if(cur_game_state == `GAME_INIT && advance_state) begin
            next_game_state = `GAME_P1_INIT_DRAW;
        end
        else if(cur_game_state == `GAME_P1_INIT_DRAW && 
                (init_draw_done || (interboard_en && interboard_msg_type == `STATE_TURN))) begin
            next_game_state = `GAME_P2_INIT_DRAW;
        end
        else if(cur_game_state == `GAME_P2_INIT_DRAW &&
                (init_draw_done || (interboard_en && interboard_msg_type == `STATE_TURN))) begin
            next_game_state = `GAME_P1_WAIT_IN;
        end
        else if(cur_game_state == `GAME_P1_WAIT_IN) begin
            if(shift_en && my_turn) begin
                next_game_state = `GAME_P1_SHIFT;
            end
            else if(valid_card_take && my_turn && l_click) begin
                next_game_state = `GAME_P1_MOVE;
            end
            else if(switch_turn) begin
                next_game_state = `GAME_P2_WAIT_IN;
            end
            else if(one_win || cheat_activate) begin
                next_game_state = `GAME_FIN;
            end
        end
        else if(cur_game_state == `GAME_P2_WAIT_IN) begin
            if(shift_en && my_turn) begin
                next_game_state = `GAME_P2_SHIFT;
            end
            else if(valid_card_take && my_turn && l_click) begin
                next_game_state = `GAME_P2_MOVE;
            end
            else if(switch_turn) begin
                next_game_state = `GAME_P2_WAIT_IN;
            end
            else if(one_win || cheat_activate) begin
                next_game_state = `GAME_FIN;
            end
        end
        else if(cur_game_state == `GAME_P1_MOVE && move_done) begin
            next_game_state = `GAME_P1_WAIT_IN;
        end
        else if(cur_game_state == `GAME_P2_MOVE && move_done) begin
            next_game_state = `GAME_P2_WAIT_IN;
        end
        else if(cur_game_state == `GAME_P1_SHIFT && (shift_done || shift_en == 0)) begin
            next_game_state = `GAME_P1_WAIT_IN;
        end
        else if(cur_game_state == `GAME_P2_SHIFT && (shift_done || shift_en == 0)) begin
            next_game_state = `GAME_P2_WAIT_IN;
        end
        else if(cur_game_state == `GAME_FIN && !advance_state) begin
            next_game_state = `GAME_FIN;
        end
        else if(cur_game_state == `GAME_FIN && advance_state) begin
            next_game_state = `GAME_INIT;
        end
    end

    always@* begin
        if(PLAYER == `P1) begin
            if(cur_game_state == `GAME_P1_WAIT_IN ||
               cur_game_state == `GAME_P1_MOVE ||
               cur_game_state == `GAME_P1_SHIFT) begin
                my_turn = 1;
            end
            else begin
                my_turn = 0;
            end
        end
        else begin
            if(cur_game_state == `GAME_P2_WAIT_IN ||
               cur_game_state == `GAME_P2_MOVE ||
               cur_game_state == `GAME_P2_SHIFT) begin
                my_turn = 1;
            end
            else begin
                my_turn = 0;
            end
        end
    end

    always@* begin
        if(resetting_table) begin
            ctrl_en = reset_table_ctrl_en;
            ctrl_move_dir = reset_table_ctrl_move_dir;
            ctrl_block_x = reset_table_ctrl_block_x;
            ctrl_block_y = reset_table_ctrl_block_y;
            ctrl_msg_type = reset_table_ctrl_msg_type;
            ctrl_card = reset_table_ctrl_card;
            ctrl_sel_len = reset_table_ctrl_sel_len;
        end
        else if(cheatting) begin
            ctrl_en = cheat_ctrl_en;
            ctrl_move_dir = cheat_ctrl_move_dir;
            ctrl_block_x = cheat_ctrl_block_x;
            ctrl_block_y = cheat_ctrl_block_y;
            ctrl_msg_type = cheat_ctrl_msg_type;
            ctrl_card = cheat_ctrl_card;
            ctrl_sel_len = cheat_ctrl_sel_len;
        end
        else if(cur_game_state == `GAME_INIT || cur_game_state == `GAME_FIN) begin
            ctrl_en = advance_state_ctrl_en;
            ctrl_move_dir = advance_state_ctrl_move_dir;
            ctrl_block_x = advance_state_ctrl_block_x;
            ctrl_block_y = advance_state_ctrl_block_y;
            ctrl_msg_type = advance_state_ctrl_msg_type;
            ctrl_card = advance_state_ctrl_card;
            ctrl_sel_len = advance_state_ctrl_sel_len;
        end
        else if(cur_game_state == `GAME_P1_INIT_DRAW || cur_game_state == `GAME_P2_INIT_DRAW) begin
            ctrl_en = init_draw_ctrl_en;
            ctrl_move_dir = init_draw_ctrl_move_dir;
            ctrl_block_x = init_draw_ctrl_block_x;
            ctrl_block_y = init_draw_ctrl_block_y;
            ctrl_msg_type = init_draw_ctrl_msg_type;
            ctrl_card = init_draw_ctrl_card;
            ctrl_sel_len = init_draw_ctrl_sel_len;
        end
        else if(cur_game_state == `GAME_P1_MOVE || cur_game_state == `GAME_P2_MOVE) begin
            ctrl_en = move_ctrl_en;
            ctrl_move_dir = move_ctrl_move_dir;
            ctrl_block_x = move_ctrl_block_x;
            ctrl_block_y = move_ctrl_block_y;
            ctrl_msg_type = move_ctrl_msg_type;
            ctrl_card = move_ctrl_card;
            ctrl_sel_len = move_ctrl_sel_len;
        end
        else if(cur_game_state == `GAME_P1_SHIFT || cur_game_state == `GAME_P2_SHIFT) begin
            // ctrl_en = shift_ctrl_en;
            // ctrl_move_dir = shift_ctrl_move_dir;
            // ctrl_block_x = shift_ctrl_block_x;
            // ctrl_block_y = shift_ctrl_block_y;
            // ctrl_msg_type = shift_ctrl_msg_type;
            // ctrl_card = shift_ctrl_card;
            // ctrl_sel_len = shift_ctrl_sel_len;
        end
        else if(cur_game_state == `GAME_P1_WAIT_IN || cur_game_state == `GAME_P2_WAIT_IN) begin
            ctrl_en = switch_turn_ctrl_en;
            ctrl_move_dir = switch_turn_ctrl_move_dir;
            ctrl_block_x = switch_turn_ctrl_block_x;
            ctrl_block_y = switch_turn_ctrl_block_y;
            ctrl_msg_type = switch_turn_ctrl_msg_type;
            ctrl_card = switch_turn_ctrl_card;
            ctrl_sel_len = switch_turn_ctrl_sel_len;
        end
        else if(my_card_cnt == 0 || oppo_card_cnt == 0) begin
            ctrl_en = one_win_ctrl_en;
            ctrl_move_dir = one_win_ctrl_move_dir;
            ctrl_block_x = one_win_ctrl_block_x;
            ctrl_block_y = one_win_ctrl_block_y;
            ctrl_msg_type = one_win_ctrl_msg_type;
            ctrl_card = one_win_ctrl_card;
            ctrl_sel_len = one_win_ctrl_sel_len;
        end
        else begin
            ctrl_en = 0;
            ctrl_move_dir = 0;
            ctrl_block_x = 0;
            ctrl_block_y = 0;
            ctrl_msg_type = 0;
            ctrl_card = 0;
            ctrl_sel_len = 0;
        end
    end

    handle_advance_state#(PLAYER) handle_advance_state(
        .clk(clk),
        .rst(rst),
        .interboard_rst(interboard_rst),

        .start_game(start_game),
        .cur_game_state(cur_game_state),
        .inter_ready(inter_ready),
        .interboard_en(interboard_en),
        .interboard_msg_type(interboard_msg_type),
        
        .advance_state(advance_state),
        
        .advance_state_ctrl_en(advance_state_ctrl_en),
        .advance_state_ctrl_move_dir(advance_state_ctrl_move_dir),
        .advance_state_ctrl_block_x(advance_state_ctrl_block_x),
        .advance_state_ctrl_block_y(advance_state_ctrl_block_y),
        .advance_state_ctrl_msg_type(advance_state_ctrl_msg_type),
        .advance_state_ctrl_card(advance_state_ctrl_card),
        .advance_state_ctrl_sel_len(advance_state_ctrl_sel_len)
    );

    handle_cheat#(PLAYER) handle_cheat(
        .clk(clk),
        .rst(rst),
        .interboard_rst(interboard_rst),

        .five_r_click(five_r_click),
        .cur_game_state(cur_game_state),
        .inter_ready(inter_ready),
        
        .interboard_en(interboard_en),
        .interboard_msg_type(interboard_msg_type),
        
        .cheat_activate(cheat_activate),
        .cheat_ctrl_en(cheat_ctrl_en),
        .cheat_ctrl_move_dir(cheat_ctrl_move_dir),
        .cheat_ctrl_block_x(cheat_ctrl_block_x),
        .cheat_ctrl_block_y(cheat_ctrl_block_y),
        .cheat_ctrl_msg_type(cheat_ctrl_msg_type),
        .cheat_ctrl_card(cheat_ctrl_card),
        .cheat_ctrl_sel_len(cheat_ctrl_sel_len)
    );

    handle_init_draw#(PLAYER) handle_init_draw(
        .clk(clk),
        .rst(rst),
        .interboard_rst(interboard_rst),

        .init_draw_en(init_draw_en),
        .cur_game_state(cur_game_state),
        .inter_ready(inter_ready),
        

        .map(map),
        .available_card(available_card),

        .init_draw_done(init_draw_done),

        .init_draw_ctrl_en(init_draw_ctrl_en),
        .init_draw_ctrl_move_dir(init_draw_ctrl_move_dir),
        .init_draw_ctrl_block_x(init_draw_ctrl_block_x),
        .init_draw_ctrl_block_y(init_draw_ctrl_block_y),
        .init_draw_ctrl_msg_type(init_draw_ctrl_msg_type),
        .init_draw_ctrl_card(init_draw_ctrl_card),
        .init_draw_ctrl_sel_len(init_draw_ctrl_sel_len)
    );

    handle_move#(PLAYER) handle_move(
        .clk(clk),
        .rst(rst),
        .interboard_rst(interboard_rst),

        // .move_en(move_en),
        .cur_game_state(cur_game_state),
        .inter_ready(inter_ready),
        .valid_card_take(valid_card_take),
        .valid_card_down(valid_card_down),
        .map(map),

        .l_click(l_click),
        .mouse_inblock(mouse_inblock),
        .mouse_block_x(mouse_block_x),
        .mouse_block_y(mouse_block_y),

        .move_done(move_done),
        .move_sel_card(move_sel_card),

        .move_ctrl_en(move_ctrl_en),
        .move_ctrl_move_dir(move_ctrl_move_dir),
        .move_ctrl_block_x(move_ctrl_block_x),
        .move_ctrl_block_y(move_ctrl_block_y),
        .move_ctrl_msg_type(move_ctrl_msg_type),
        .move_ctrl_card(move_ctrl_card),
        .move_ctrl_sel_len(move_ctrl_sel_len)
    );

    handle_one_win#(PLAYER) handle_one_win(
        .clk(clk),
        .rst(rst),
        .interboard_rst(interboard_rst),

        .done_and_next(done_and_next),
        .my_card_cnt(my_card_cnt),
        .cur_game_state(cur_game_state),
        .inter_ready(inter_ready),

        .interboard_en(interboard_en),
        .interboard_msg_type(interboard_msg_type),
        
        .one_win(one_win),
        
        .one_win_ctrl_en(one_win_ctrl_en),
        .one_win_ctrl_move_dir(one_win_ctrl_move_dir),
        .one_win_ctrl_block_x(one_win_ctrl_block_x),
        .one_win_ctrl_block_y(one_win_ctrl_block_y),
        .one_win_ctrl_msg_type(one_win_ctrl_msg_type),
        .one_win_ctrl_card(one_win_ctrl_card),
        .one_win_ctrl_sel_len(one_win_ctrl_sel_len)
    );

    handle_reset_table#(PLAYER) handle_reset_table(
        .clk(clk),
        .rst(rst),
        .interboard_rst(interboard_rst),

        .reset_table_raw(reset_table_raw),
        .cur_game_state(cur_game_state),
        .inter_ready(inter_ready),
        
        .interboard_en(interboard_en),
        .interboard_msg_type(interboard_msg_type),
        
        .reset_table(reset_table),
        
        .reset_table_ctrl_en(reset_table_ctrl_en),
        .reset_table_ctrl_move_dir(reset_table_ctrl_move_dir),
        .reset_table_ctrl_block_x(reset_table_ctrl_block_x),
        .reset_table_ctrl_block_y(reset_table_ctrl_block_y),
        .reset_table_ctrl_msg_type(reset_table_ctrl_msg_type),
        .reset_table_ctrl_card(reset_table_ctrl_card),
        .reset_table_ctrl_sel_len(reset_table_ctrl_sel_len)
    );

    // handle_shift#(PLAYER) handle_shift(
    //     .clk(clk),
    //     .rst(rst),
    //     .interboard_rst(interboard_rst),
        
    //     .move_left(move_left),
    //     .move_right(move_right),
    //     .shift_en(shift_en),
    //     .cur_game_state(cur_game_state),
    //     .inter_ready(inter_ready),

    //     .shift_done(shift_done),
    //     .shift_sel_card(shift_sel_card),
        
    //     .shift_ctrl_en(shift_ctrl_en),
    //     .shift_ctrl_move_dir(shift_ctrl_move_dir),
    //     .shift_ctrl_block_x(shift_ctrl_block_x),
    //     .shift_ctrl_block_y(shift_ctrl_block_y),
    //     .shift_ctrl_msg_type(shift_ctrl_msg_type),
    //     .shift_ctrl_card(shift_ctrl_card),
    //     .shift_ctrl_sel_len(shift_ctrl_sel_len)
    // );

    handle_switch_turn#(PLAYER) handle_switch_turn(
        .clk(clk),
        .rst(rst),
        .interboard_rst(interboard_rst),

        .done_and_next(done_and_next),
        .draw_and_next(draw_and_next),
        .can_done(can_done),
        .can_draw(can_draw),
        .cur_game_state(cur_game_state),
        .inter_ready(inter_ready),
        .map(map),
        .available_card(available_card),
        .my_card_cnt(my_card_cnt),
        
        .interboard_en(interboard_en),
        .interboard_msg_type(interboard_msg_type),
        
        .switch_turn(switch_turn),
        
        .switch_turn_ctrl_en(switch_turn_ctrl_en),
        .switch_turn_ctrl_move_dir(switch_turn_ctrl_move_dir),
        .switch_turn_ctrl_block_x(switch_turn_ctrl_block_x),
        .switch_turn_ctrl_block_y(switch_turn_ctrl_block_y),
        .switch_turn_ctrl_msg_type(switch_turn_ctrl_msg_type),
        .switch_turn_ctrl_card(switch_turn_ctrl_card),
        .switch_turn_ctrl_sel_len(switch_turn_ctrl_sel_len)
    );
endmodule


