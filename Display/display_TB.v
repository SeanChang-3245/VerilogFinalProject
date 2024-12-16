module Display_Test(
	input wire clk, 
	input wire rst,
	input wire change_pat,
	input wire change_sel,
	inout PS2_CLK, 
	inout PS2_DATA,

    output wire hsync,
    output wire vsync,
    output wire [3:0] vgaRed,
    output wire [3:0] vgaGreen,
    output wire [3:0] vgaBlue,
	output wire [6:0] DISPLAY,
	output wire [3:0] DIGIT
);	
	wire [9:0] h_cnt;
	wire [9:0] v_cnt;
	wire [11:0] mouse_pixel;
	wire en_mouse_display;
	reg [8*18-1:0] sel_card, next_sel_card;
	reg [8*18*6-1:0] map, next_map;

	wire change_pat_button;
	button_preprocess btn_pre_inst0( .clk(clk), .signal_in(change_pat), .signal_out(change_pat_button));

	wire change_sel_button;
	button_preprocess btn_pre_inst1( .clk(clk), .signal_in(change_sel), .signal_out(change_sel_button));

	MouseInterface_top MouseIn_tp_inst0(
		.clk(clk),
		.rst(rst),
		.interboard_rst(0),
		.h_cnt(h_cnt),
		.v_cnt(v_cnt),
		.PS2_CLK(PS2_CLK),
		.PS2_DATA(PS2_DATA),
		.en_mouse_display(en_mouse_display),
		.mouse_pixel(mouse_pixel)
	);
	Display_top display_inst0(
		.clk(clk),
		.rst(rst),
		.interboard_rst(0),
		.en_mouse_display(en_mouse_display),
		.oppo_card_cnt(20),
		.deck_card_cnt(30),
		.mouse_pixel(mouse_pixel),
		.sel_card(sel_card),
		.map(map),
		.h_cnt(h_cnt),
		.v_cnt(v_cnt),
		.hsync(hsync),
		.vsync(vsync),
		.vgaRed(vgaRed), 
		.vgaGreen(vgaGreen), 
		.vgaBlue(vgaBlue),
		.DISPLAY(DISPLAY),
		.DIGIT(DIGIT)
	);

	
	always @(*) begin
		next_map = map;
		if(change_pat_button) begin
			next_map = map >> 12;
			next_map[863 -: 12] = {6'd53, 6'd52};
		end
	end
	always @(*) begin
		next_sel_card = sel_card;
		if(change_sel_button) begin
			next_sel_card = ~sel_card;
		end
	end
	integer i, j;
	always @(posedge clk) begin
		if(rst) begin
			for(i = 0; i < 144; i = i + 1) begin
				map[i*6+5 -:6] <= i%55;
				sel_card[i] <= 1;
			end
		end else begin
			map <= next_map;
			sel_card <= next_sel_card;
		end
	end
endmodule