module Display_top(
    input wire clk,
    input wire rst,
    input wire interboard_rst,
    input wire en_mouse_display,
    input wire [6:0] oppo_card_cnt,	// display on sevensegment rigth two digits
    input wire [6:0] deck_card_cnt,	// display on sevensegment left two digits
    input wire [11:0] mouse_pixel,
    input wire [8*18-1:0] sel_card,
    input wire [8*18*6-1:0] map,	// 8*18 positions with 54 different types of cards

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
    clock_divider #(.n(2)) m2 (.clk(clk), .clk_div(clk_25MHz));

	wire all_rst;
	assign all_rst = rst | interboard_rst;
	reg [11:0] pixel;
	wire [11:0] card_pixel;
	wire [15:0] nums;
	wire [9:0] vga_h_cnt, vga_v_cnt;
	assign {h_cnt, v_cnt} = {vga_h_cnt, vga_v_cnt};
	assign {vgaRed, vgaGreen, vgaBlue} = (valid) ? pixel : 12'h0;
    vga_controller vga_inst(
        .pclk(clk_25MHz),
        .reset(all_rst),
        .hsync(hsync),
        .vsync(vsync),
        .valid(valid),
        .h_cnt(vga_h_cnt),
        .v_cnt(vga_v_cnt)
    );

	Display_card draw_card_inst(
		.clk(clk),
		.clk_25MHz(clk_25MHz),
		.rst(all_rst),
		.map(map),
		.sel_card(sel_card),
		.h_cnt(vga_h_cnt),
		.v_cnt(vga_v_cnt),
		.card_pixel(card_pixel)
	);
	
	SevenSegment Sevenseg_inst0(
		.clk(clk), 
		.rst(all_rst), 
		.nums(nums),
		.display(DISPLAY),
		.digit(DIGIT)
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
    // else begin
    //     pixel = bg_pixel
    // end

	always @(*) begin
		if(en_mouse_display) begin
			pixel = mouse_pixel;
		end
		else if(card_valid) begin
			pixel = card_pixel;
		end
		else begin
			pixel = 12'h68A;
		end
	end
	// card_valid control
	assign card_valid = (h_cnt >= 32 && h_cnt < 608) && 
						((v_cnt >= 19 && v_cnt < 65) || (v_cnt >= 74 && v_cnt < 120) ||
						 (v_cnt >= 129 && v_cnt < 175) || (v_cnt >= 184 && v_cnt < 230)||
						 (v_cnt >= 239 && v_cnt < 285) || (v_cnt >= 294 && v_cnt < 340)||
						 (v_cnt >= 360 && v_cnt < 406) || (v_cnt >= 415 && v_cnt < 461));

	wire [3:0] deck_one, oppo_one;

	mod_ten mod_ten_inst0(
		.in(deck_card_cnt),
		.out(deck_one)
	);

	mod_ten mod_ten_inst1(
		.in(oppo_card_cnt),
		.out(oppo_one)
	);

	wire [3:0] deck_ten = deck_card_cnt/10;
	// wire [3:0] deck_one = deck_card_cnt%10;
	wire [3:0] oppo_ten = oppo_card_cnt/10;
	// wire [3:0] oppo_one = oppo_card_cnt%10;
	assign nums = {oppo_ten, oppo_one, deck_ten, deck_one};
	
//	ila_0 ila_inst (clk, h_cnt, v_cnt, card_pixel, draw_card_inst.mem_pixel_inst.pixel_addr, draw_card_inst.mem_pixel_inst.pixel_x, draw_card_inst.mem_pixel_inst.pixel_y);			

endmodule


module mod_ten(
	input wire [6:0] in,
	output reg [3:0] out
);

	always @(*) begin
		if(in < 10) out = in;
		else if (in < 20) out = in - 10;
		else if (in < 30) out = in - 20;
		else if (in < 40) out = in - 30;
		else if (in < 50) out = in - 40;
		else if (in < 60) out = in - 50;
		else if (in < 70) out = in - 60;
		else if (in < 80) out = in - 70;
		else if (in < 90) out = in - 80;
		else if (in < 100) out = in - 90;
		else out = in - 100;
	end

endmodule