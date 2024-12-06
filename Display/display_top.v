module Display_top(
    input wire clk,
    input wire rst,
    input wire interboard_rst,
    input wire en_mouse_display,
    input wire [6:0] oppo_card_cnt,
    input wire [6:0] deck_card_cnt,
    input wire [11:0] mouse_pixel,
    input wire [8*18-1:0] sel_card,
    input wire [8*18*6-1:0] map,

    output wire hsync,
    output wire vsync,
    output wire [9:0] h_cnt,
    output wire [9:0] v_cnt,
    output wire [3:0] vgaRed,
    output wire [3:0] vgaGreen,
    output wire [3:0] vgaBlue,
    output wire [6:0] DISPLAY,
    output wire [3:0] DIGIT
);

    wire clk_25MHz;
    clock_divider m2 (.clk(clk), .clk_div(clk_25MHz));


    vga_controller vga_inst(
        .pclk(clk_25MHz),
        .reset(rst),
        .hsync(hsync),
        .vsync(vsync),
        .valid(valid),
        .h_cnt(h_cnt),
        .v_cnt(v_cnt)
    );


    // use if-else to determine which object should be drawn on the top
    // each object should output a signal to indicate whether is should be drawn at this pixel
    // priority: mouse > card = button > background
    // e.g.
    // if(mouse_valid) begin
    //     pixel = mouse_pixel
    // end
    // else if(card_valid) begin
    //     pixel = card_pixel
    // end
    // else if(button_valid) begin
    //     pixel = button_pixel
    // end
    // else begin
    //     pixel = bg_pixel
    // end

	// mouse inst ========================================================================
	// 


endmodule