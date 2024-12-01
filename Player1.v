module P1_top (
    input wire clk,
    input wire rst,
    input wire btnL,
    input wire btnR,
    input wire [15:0] SW,
    
    inout request,
    inout ack,
    inout [5:0] interboard_data,
    inout wire PS2_CLK,          // PS2 Mouse
    inout wire PS2_DATA,         // PS2 Mouse

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

    // 所有 rst 都要換成 rstGame
    // rstGame 包含這張板子的 rst 和從interboard來的 rst
    wire rstGame = rst;

    // Preprocess button and switch
    wire reset_table;
    wire done_and_next;
    wire draw_and_next;
    wire move_right, move_left;
    wire start_game;

    // Display output
    // hsync, vsync, vgaRed, vgaGreen, vgaBlue
    // directly connect to final output

    // MouseInterface output
    wire mouse_valid;
    wire l_click;
    wire cheat_activate;
    wire [9:0] mouse_x;
    wire [8:0] mouse_y;
    wire [4:0] mouse_block_x;
    wire [2:0] mouse_block_y;

    // RuleCheck output
    wire rule_valid;

    // InterboardCommunication output
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

    // GameControl output
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
        .rst(rstGame),
        .PS2_CLK(PS2_CLK),
        .PS2_DATA(PS2_DATA),
        .mouse_valid(mouse_valid),
        .l_click(l_click),
        .cheat_activate(cheat_activate),
        .mouse_x(mouse_x),
        .mouse_y(mouse_y),
        .mouse_block_x(mouse_block_x),
        .mouse_block_y(mouse_block_y)
    );

    GameControl_top  #(.PLAYER(PLAYER)) game_control_top_inst(
        .clk(clk),
        .rst(rstGame),
        .start_game(start_game),
        .rule_valid(rule_valid),
        .mouse_valid(mouse_valid),
        .cheat_activate(cheat_activate),
        .move_left(move_left),
        .move_right(move_right),
        .reset_table(reset_table),
        .done_and_next(done_and_next),
        .draw_and_next(draw_and_next),
        .available_card(available_card),
        .picked_card(picked_card),
        .mouse_x(mouse_x),
        .mouse_y(mouse_y),
        .mouse_block_x(mouse_block_x),
        .mouse_block_y(mouse_block_y),

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
        .rst(rstGame),
        .mouse_x(mouse_x),
        .mouse_y(mouse_y),
        .sel_card(sel_card),
        .map(map),

        .vgaRed(vgaRed),
        .vgaGreen(vgaGreen),
        .vgaBlue(vgaBlue),
        .hsync(hsync),
        .vsync(vsync)
    );

    InterboardCommunication_top interboard_communication_top_inst(
        .clk(clk),
        .rst(rstGame),  // maye need to change to rst
        .ctrl_en(ctrl_en),
        .ctrl_move_dir(ctrl_move_dir),
        .ctrl_block_x(ctrl_block_x),
        .ctrl_block_y(ctrl_block_y),
        .ctrl_msg_type(ctrl_msg_type),
        .ctrl_card(ctrl_card),
        .ctrl_sel_len(ctrl_sel_len),
        
        .request(request),
        .ack(ack),
        .interboard_data(interboard_data),

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
        .rst(rstGame),
        .map(map),
        .rule_valid(rule_valid)
    );

    MemoryHandle_top memory_handle_top_inst(
        .clk(clk),
        .rst(rstGame),

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
        .map(map)
    );
   
endmodule