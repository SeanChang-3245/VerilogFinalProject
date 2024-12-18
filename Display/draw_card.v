module Draw_card(
	input wire clk,
	input wire clk_25MHz,
	input wire rst,
	input wire [8*18*6-1:0] map,
	input wire [8*18-1:0] sel_card,
	input wire [9:0] h_cnt,
	input wire [9:0] v_cnt,
	output reg [11:0] card_pixel
);

	localparam FRAME_COLOR = 12'hFEC;


	reg [5:0] card_type;
	wire [11:0] mem_card_pixel;
	reg [5:0] pixel_x;
	reg [5:0] pixel_y;
	Mem_pixel mem_pixel_inst(
		.clk_25MHz(clk_25MHz),
		.pixel_x(pixel_x),
		.pixel_y(pixel_y),
		.card_type(card_type),
		.card_pixel(mem_card_pixel)
	);

	reg [5:0] card_table [0:8*18-1];
	reg selected_card [0:8*18-1];
	reg [4:0] x;
	reg [2:0] y;
	wire [7:0] position;


	integer i, j;
	always @(*) begin
		for(i = 0; i < 144; i = i + 1) begin
			card_table[i] = map[i*6+5 -: 6];
		end
	end
	always @(*) begin
		for(j = 0; j < 144; j = j + 1) begin
			selected_card[j] = sel_card[j];
		end
	end

	// CARD XY GEN
	assign position = (x + y*18);
	always @(*) begin
		x = 0;
		if(h_cnt >= 32 && h_cnt < 607) begin
			x = (h_cnt-32)/32;
		end
	end
	// (v_cnt >= 19 && v_cnt < 65) || (v_cnt >= 74 && v_cnt < 120) ||
						//  (v_cnt >= 129 && v_cnt < 175) || (v_cnt >= 184 && v_cnt < 230)||
						//  (v_cnt >= 239 && v_cnt < 285) || (v_cnt >= 294 && v_cnt < 340)||
						//  (v_cnt >= 360 && v_cnt < 406) || (v_cnt >= 415 && v_cnt < 461)
	always @(*) begin
		if(v_cnt >= 19 && v_cnt < 65) begin
			y = 0;
		end
		else if(v_cnt >= 74 && v_cnt < 120) begin
			y = 1;
		end
		else if(v_cnt >= 129 && v_cnt < 175) begin
			y = 2;
		end
		else if(v_cnt >= 184 && v_cnt < 230) begin
			y = 3;
		end
		else if(v_cnt >= 239 && v_cnt < 285) begin
			y = 4;
		end
		else if(v_cnt >= 294 && v_cnt < 340) begin
			y = 5;
		end
		else if(v_cnt >= 360 && v_cnt < 406) begin
			y = 6;
		end
		else if(v_cnt >= 415 && v_cnt < 461) begin
			y = 7;
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
	// decide card frame
	always @(*) begin
		if(selected_card[position] && card_table[position]!=54) begin
			if(pixel_y < 2 || pixel_y >= 44 && pixel_y < 46) begin
				card_pixel = FRAME_COLOR;
			end
			else if(pixel_x < 2 || pixel_x >= 30 && pixel_x < 32)begin
				card_pixel = FRAME_COLOR;
			end else begin
				card_pixel = mem_card_pixel;
			end
		end else begin
			card_pixel = mem_card_pixel;
		end
	end
endmodule