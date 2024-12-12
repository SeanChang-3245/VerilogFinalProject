module Memory_Test(
	input wire clk,
	input wire rst,
	inout PS2_CLK, 
	inout PS2_DATA,

	output wire hsync,
    output wire vsync,
    output wire [3:0] vgaRed,
    output wire [3:0] vgaGreen,
    output wire [3:0] vgaBlue
);	
	wire [9:0] h_cnt;
	wire [9:0] v_cnt;
	wire valid;
	wire [11:0] pixel;
	reg [14:0] pixel_addr;
	assign {vgaRed, vgaGreen, vgaBlue} = (valid) ? pixel : 12'h0;
	wire clk_25MHz;
    clock_divider #(.n(2)) m2 (.clk(clk), .clk_div(clk_25MHz));

	vga_controller vga_inst(
		.pclk(clk_25MHz),
		.reset(rst),
		.hsync(hsync),
		.vsync(vsync),
		.valid(valid),
		.h_cnt(h_cnt),
		.v_cnt(v_cnt)
    );
	blk_mem_gen_6 blk_mem_gen_6_inst( .clka(clk_25MHz), .dina(dina), .wea(0), .addra(pixel_addr), .douta(pixel));
	wire [3:0] card_num;
	reg [5:0] pixel_x;
	reg [5:0] pixel_y;
	assign card_num = h_cnt/32;
	
	always @(*) begin
		pixel_x = h_cnt%32;
	end
	always @(*) begin
		pixel_y = v_cnt%46;
	end
	always @(*) begin
		pixel_addr = 32*46*5 + 32*pixel_y + pixel_x;
	end
endmodule