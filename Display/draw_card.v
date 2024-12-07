module Draw_card(
	input wire clk,
	input wire clk_25MHz,
	input wire rst,
	input [8*18*6-1:0] map,
	input wire [9:0] h_cnt,
	input wire [9:0] v_cnt,
	output reg [11:0] card_pixel
);

	reg [5:0] card_type;
	reg [5:0] pixel_x;
	reg [5:0] pixel_y;
	mem_pixel_gen mem_pixel_gen_inst(
		.clk(clk_25MHz),
		.pixel_x(pixel_x),
		.pixel_y(pixel_y),
		.card_type(card_type),
		.card_pixel(card_pixel)
	);

	reg [5:0] card_table [0:8*18-1];
	reg [4:0] x;
	reg [2:0] y;
	wire [7:0] position;


	integer i;
	always @(*) begin
		for(i = 0; i < 144; i = i + 1) begin
			card_table[i] = map[i*6+5 -: 6];
		end
	end

	// CARD XY GEN
	assign position = (x*18 + y);
	always @(*) begin
		x = 0;
		if(h_cnt >= 32 && h_cnt < 607) begin
			x = (h_cnt-32)/32;
		end
	end
	always @(*) begin
		if(v_cnt >= 19 && v_cnt < 349) begin
			y = (v_cnt-19)/55;
		end
		else if(v_cnt >= 360 && v_cnt < 461) begin
			y = 6 + (v_cnt-360)/55;
		end 
		else begin
			y = 0;
		end
	end
	// CARD TYPE GEN
	always @(*) begin
		card_type = card_table[position];
	end
	// PIXEL X GEN
	always @(*) begin
		pixel_x = 0;
		if(h_cnt >= 32 && h_cnt < 607) begin
			pixel_x = h_cnt - 32 - x*32;
		end
	end
	// PIXEL Y GEN
	always @(*) begin
		pixel_y = 0;
		if(v_cnt >= 19 && v_cnt < 349) begin
			pixel_y = v_cnt - 19 - y*55;
		end
		else if(v_cnt >= 360 && v_cnt < 461) begin
			pixel_y = v_cnt - 360 - (y-6)*55;
		end
	end
endmodule