module mouse(
    input clk,
	input rst,
    input [9:0] h_cntr_reg,
    input [9:0] v_cntr_reg,
    output enable_mouse_display,
    output [9:0] MOUSE_X_POS,
    output [9:0] MOUSE_Y_POS,
    output MOUSE_LEFT,
    output MOUSE_RIGHT,
	output wire [11:0] mouse_pixel,
    inout PS2_CLK,
    inout PS2_DATA
);

    wire [3:0] MOUSE_Z_POS;
	wire MOUSE_MIDDLE, MOUSE_NEW_EVENT;
	wire [3:0] mouse_cursor_red;
	wire [3:0] mouse_cursor_green;
	wire [3:0] mouse_cursor_blue;
    assign mouse_pixel = {mouse_cursor_red, mouse_cursor_green, mouse_cursor_blue};
	
    MouseCtl #(
      .SYSCLK_FREQUENCY_HZ(108000000),
      .CHECK_PERIOD_MS(500),
      .TIMEOUT_PERIOD_MS(100)
    )MC1(
        .clk(clk),
        .rst(rst),
        .xpos(MOUSE_X_POS),
        .ypos(MOUSE_Y_POS),
        .zpos(MOUSE_Z_POS),
        .left(MOUSE_LEFT),
        .middle(MOUSE_MIDDLE),
        .right(MOUSE_RIGHT),
        .new_event(MOUSE_NEW_EVENT),
        .value(12'd0),
        .setx(1'b0),
        .sety(1'b0),
        .setmax_x(1'b0),
        .setmax_y(1'b0),
        .ps2_clk(PS2_CLK),
        .ps2_data(PS2_DATA)
    );
    
    MouseDisplay MD1(
        .pixel_clk(clk),
        .xpos(MOUSE_X_POS),
        .ypos(MOUSE_Y_POS),
        .hcount(h_cntr_reg),
        .vcount(v_cntr_reg),
        .enable_mouse_display_out(enable_mouse_display),
        .red_out(mouse_cursor_red),
        .green_out(mouse_cursor_green),
        .blue_out(mouse_cursor_blue)
    );

endmodule