module Mem_pixel(
	input wire clk_25MHz,
	input wire [5:0] pixel_x,
	input wire [5:0] pixel_y,
	input wire [5:0] card_type,
	output reg [11:0] card_pixel
);
	wire [11:0] num_pixel [0:3];
	wire [11:0] face_pixel [0:1];
	reg [14:0] pixel_addr;
	// 0: black, 1: blue, 2: red, 3: orange
	blk_mem_gen_0 blk_mem_gen_0_inst( .clka(clk_25MHz), .wea(0), .addra(pixel_addr), .douta(num_pixel[0]));
	blk_mem_gen_1 blk_mem_gen_1_inst( .clka(clk_25MHz), .wea(0), .addra(pixel_addr), .douta(num_pixel[1]));
	blk_mem_gen_2 blk_mem_gen_2_inst( .clka(clk_25MHz), .wea(0), .addra(pixel_addr), .douta(num_pixel[2]));
	blk_mem_gen_3 blk_mem_gen_3_inst( .clka(clk_25MHz), .wea(0), .addra(pixel_addr), .douta(num_pixel[3]));
	// 0: red_face, 1: black_face
	blk_mem_gen_4 blk_mem_gen_4_inst( .clka(clk_25MHz), .wea(0), .addra(pixel_addr), .douta(face_pixel[0]));
	blk_mem_gen_5 blk_mem_gen_5_inst( .clka(clk_25MHz), .wea(0), .addra(pixel_addr), .douta(face_pixel[1]));

	always @(*) begin
		pixel_addr = 0;
		if(card_type < 13) begin
			pixel_addr = card_type*32*46 + pixel_y*32 + pixel_x;
		end else if(card_type < 26) begin
			pixel_addr = (card_type-13)*32*46 + pixel_y*32 + pixel_x;
		end else if(card_type < 39) begin
			pixel_addr = (card_type-26)*32*46 + pixel_y*32 + pixel_x;
		end else if(card_type < 52) begin
			pixel_addr = (card_type-39)*32*46 + pixel_y*32 + pixel_x;
		end else if(card_type == 52) begin
			pixel_addr = pixel_y*32 + pixel_x;
		end else if(card_type == 53) begin
			pixel_addr = pixel_y*32 + pixel_x;
		end
	end

	always @(*) begin
		card_pixel = 0;
		if(card_type < 13) begin
			card_pixel = num_pixel[0];
		end else if(card_type < 26) begin
			card_pixel = num_pixel[1];
		end else if(card_type < 39) begin
			card_pixel = num_pixel[2];
		end else if(card_type < 52) begin
			card_pixel = num_pixel[3];
		end else if(card_type == 52) begin
			card_pixel = face_pixel[0];
		end else if(card_type == 53) begin
			card_pixel = face_pixel[1];
		end else begin
			card_pixel = 12'h68A;
		end
	end
endmodule