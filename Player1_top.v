module P1_top (
    input wire clk,
    input wire rst,
    input wire btnL,
    input wire btnR,
    input wire Request_in,
    input wire Ack_in,
    input wire [5:0] inter_data_in,
    input wire [15:0] SW,
    
    inout wire PS2_CLK,          // PS2 Mouse
    inout wire PS2_DATA,         // PS2 Mouse

    output wire Request_out,
    output wire Ack_out,
    output wire [5:0] inter_data_out,
    output wire can_done,        // LED[1]
    output wire can_draw,        // LED[2]
    output wire [3:0] DIGIT,     // 7-segment display
    output wire [6:0] DISPLAY,   // 7-segment display
    output wire [3:0] vgaRed,    // VGA
    output wire [3:0] vgaGreen,  // VGA
    output wire [3:0] vgaBlue,   // VGA
    output wire hsync,           // VGA
    output wire vsync            // VGA
);
    // Game Control 要把這個用 parameter 傳進去 
    localparam PLAYER = 0; 

    // Preprocess button and switch
    wire reset_table;
    wire done_and_next;
    wire draw_and_next;
    wire move_right, move_left;
    wire start_game;
    wire shift_en = SW[5];

    button_preprocess bp1(.clk(clk), .signal_in(SW[15]), .signal_out(reset_table));
    button_preprocess bp2(.clk(clk), .signal_in(SW[0]), .signal_out(start_game));
    button_preprocess bp3(.clk(clk), .signal_in(SW[1]), .signal_out(done_and_next));
    button_preprocess bp4(.clk(clk), .signal_in(SW[2]), .signal_out(draw_and_next));
    button_preprocess bp5(.clk(clk), .signal_in(btnL), .signal_out(move_left));
    button_preprocess bp6(.clk(clk), .signal_in(btnR), .signal_out(move_right));


    // Display output
    // hsync, vsync, vgaRed, vgaGreen, vgaBlue, DIGIT, DISPLAY
    // directly connect to final output
    wire [9:0] h_cnt, v_cnt;

    // MouseInterface output
    wire mouse_inblock;
    wire en_mouse_display;
    wire l_click;
    wire cheat_activate;
    wire [11:0] mouse_pixel;
    wire [9:0] mouse_x;
    wire [8:0] mouse_y;
    wire [4:0] mouse_block_x;
    wire [2:0] mouse_block_y;

    // RuleCheck output
    wire rule_valid;

    // InterboardCommunication output
    wire send_ready;
    wire interboard_rst;
    wire interboard_en;
    wire interboard_move_dir;
    wire [4:0] interboard_block_x;
    wire [2:0] interboard_block_y;
    wire [3:0] interboard_msg_type;
    wire [5:0] interboard_card;
    wire [2:0] interboard_sel_len;

    // MemoryHandle output
    wire [5:0] picked_card;
    wire [105:0] available_card;
    wire [8*18*6-1:0] map;
    wire [6:0] oppo_card_cnt;
    wire [6:0] deck_card_cnt;

    // GameControl output
    // can_done, can_draw are directly connected to final output
    wire transmit;
    wire ctrl_en;
    wire ctrl_move_dir;
    wire [4:0] ctrl_block_x;
    wire [2:0] ctrl_block_y;
    wire [3:0] ctrl_msg_type;
    wire [5:0] ctrl_card;
    wire [2:0] ctrl_sel_len;
    wire [8*18-1:0] sel_card;

    MouseInterface_top mouse_interface_top_inst(
        .clk(clk),
        .rst(rst),
        .interboard_rst(interboard_rst),
        .h_cnt(h_cnt),
        .v_cnt(v_cnt),

        .PS2_CLK(PS2_CLK),
        .PS2_DATA(PS2_DATA),
        
        .mouse_inblock(mouse_inblock),
        .en_mouse_display(en_mouse_display),
        .l_click(l_click),
        .cheat_activate(cheat_activate),
        .mouse_pixel(mouse_pixel),
        .mouse_x(mouse_x),
        .mouse_y(mouse_y),
        .mouse_block_x(mouse_block_x),
        .mouse_block_y(mouse_block_y)
    );

    GameControl_top  #(.PLAYER(PLAYER)) game_control_top_inst(
        .clk(clk),
        .rst(rst),
        .interboard_rst(interboard_rst),
        .shift_en(shift_en),
        .start_game(start_game),
        .rule_valid(rule_valid),
        .mouse_inblock(mouse_inblock),
        .cheat_activate(cheat_activate),
        .move_left(move_left),
        .move_right(move_right),
        .reset_table(reset_table),
        .done_and_next(done_and_next),
        .draw_and_next(draw_and_next),
        .interboard_en(interboard_en),
        .interboard_msg_type(interboard_msg_type),
        .available_card(available_card),
        .picked_card(picked_card),
        .mouse_x(mouse_x),
        .mouse_y(mouse_y),
        .mouse_block_x(mouse_block_x),
        .mouse_block_y(mouse_block_y),

        .can_done(can_done),
        .can_draw(can_draw),
        .transmit(transmit),
        .ctrl_en(ctrl_en),
        .ctrl_move_dir(ctrl_move_dir),
        .ctrl_block_x(ctrl_block_x),
        .ctrl_block_y(ctrl_block_y),
        .ctrl_msg_type(ctrl_msg_type),
        .ctrl_card(ctrl_card),
        .ctrl_sel_len(ctrl_sel_len),

        .sel_card(sel_card)
    );

    Display_top display_top_inst(
        .clk(clk),
        .rst(rst),
        .interboard_rst(interboard_rst),
        .en_mouse_display(en_mouse_display),
        .oppo_card_cnt(oppo_card_cnt),
        .deck_card_cnt(deck_card_cnt),
        .mouse_pixel(mouse_pixel),
        .sel_card(sel_card),
        .map(map),

        .hsync(hsync),
        .vsync(vsync),
        .h_cnt(h_cnt),
        .v_cnt(v_cnt),
        .vgaRed(vgaRed),
        .vgaGreen(vgaGreen),
        .vgaBlue(vgaBlue),
        .DISPLAY(DISPLAY),
        .DIGIT(DIGIT)
    );

    InterboardCommunication_top interboard_communication_top_inst(
        .clk(clk),
        .rst(rst), 
        .transmit(transmit),
        .Request_in(Request_in),
        .Ack_in(Ack_in),
        .inter_data_in(inter_data_in),
        .ctrl_en(ctrl_en),
        .ctrl_move_dir(ctrl_move_dir),
        .ctrl_block_x(ctrl_block_x),
        .ctrl_block_y(ctrl_block_y),
        .ctrl_msg_type(ctrl_msg_type),
        .ctrl_card(ctrl_card),
        .ctrl_sel_len(ctrl_sel_len),        

        .send_ready(send_ready),
        .Request_out(Request_out),
        .Ack_out(Ack_out),
        .inter_data_out(inter_data_out),
        .interboard_rst(interboard_rst),
        .interboard_en(interboard_en),
        .interboard_move_dir(interboard_move_dir),
        .interboard_block_x(interboard_block_x),
        .interboard_block_y(interboard_block_y),
        .interboard_msg_type(interboard_msg_type),
        .interboard_card(interboard_card),
        .interboard_sel_len(interboard_sel_len)
    );

    RuleCheck_top rule_check_top_inst(
        .clk(clk),
        .rst(rst),
        .interboard_rst(interboard_rst),
        .map(map),
        .rule_valid(rule_valid)
    );

    MemoryHandle_top memory_handle_top_inst(
        .clk(clk),
        .rst(rst),
        .interboard_rst(interboard_rst),

        .transmit(transmit),
        .ctrl_en(ctrl_en),
        .ctrl_move_dir(ctrl_move_dir),
        .ctrl_msg_type(ctrl_msg_type),
        .ctrl_block_x(ctrl_block_x),
        .ctrl_block_y(ctrl_block_y),
        .ctrl_card(ctrl_card),
        .ctrl_sel_len(ctrl_sel_len),

        .interboard_en(interboard_en),
        .interboard_move_dir(interboard_move_dir),
        .interboard_msg_type(interboard_msg_type),
        .interboard_block_x(interboard_block_x),
        .interboard_block_y(interboard_block_y),
        .interboard_card(interboard_card),
        .interboard_sel_len(interboard_sel_len),

        .picked_card(picked_card),
        .available_card(available_card),

        .oppo_card_cnt(oppo_card_cnt),
        .deck_card_cnt(deck_card_cnt),
        .map(map)
    );
   
endmodule