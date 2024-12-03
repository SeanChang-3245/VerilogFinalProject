module MouseInterface_top (
    input wire clk,
    input wire rst,
    
    inout wire PS2_CLK,          // PS2 Mouse
    inout wire PS2_DATA,         // PS2 Mouse
    
    output wire mouse_inblock,     // 處理滑鼠不在任何一個區塊的情況
    output wire l_click,
    output wire cheat_activate,
    output wire [9:0] mouse_x,		// position on screen
    output wire [8:0] mouse_y,
    output wire [4:0] mouse_block_x,	// mouse in which block
    output wire [2:0] mouse_block_y
);

	wire [7:0] x, y;
	wire x_ov, x_sign, y_ov, y_sign;
	wire o_r_click, o_l_click, valid;

	ps2_mouse m1(
		.i_clk(clk), i_rst(rst), .i_PS2Clk(PS2_CLK), .i_PS2Data(PS2_DATA),
		.o_x(x), .o_x_ov(x_ov), .o_x_sign(x_sign), 
		.o_y(y), .o_y_ov(y_ov), .o_y_sign(y_sign),
		.o_r_click(o_r_click), .o_l_click(o_l_click), .o_valid(valid)
	);

	parameter NONE = 0;
	parameter RIGHT = 1;
	parameter LEFT = 2;

	parameter SPEED = 1;

	reg [1:0] cur_click;
	reg [2:0] countR, countR_next;
	reg [2:0] countL, countL_next;
	reg [4:0] mouse_block_x_next;
	reg [2:0] mouse_block_y_next;

	assign cheat_activate = ((cur_click == RIGHT) && (countR == 4)) ? 1 : 0;
	assign l_click = cur_click == LEFT ? 1 : 0;

	// handle cur_click
	always @(*) begin
		cur_click = NONE;
		if(valid) begin
			if(o_r_click) cur_click = RIGHT;
			else if(o_l_click) cur_click = LEFT;
		end
	end

	// continuous click count
	always @(*) begin
		countR_next = countR;
		countL_next = countL;
		if(cur_click == LEFT) begin
			countL_next = countL + 1;
			countR_next = 0;
		end
		else if(cur_click == RIGHT) begin
			countR_next = countR + 1;
			countL_next = 0;
		end
	end

	always @(*) begin
		mouse_block_x_next = mouse_block_x;
		mouse_block_y_next = mouse_block_y;
		if(valid) begin
			if(x_ov) begin
				mouse_block_x_next <= (x_sign) ? mouse_block_x + SPEED * x : mouse_block_x - SPEED * x;
			end
			else begin
				mouse_block_x_next <= (x_sign) ? mouse_block_x - SPEED * x : mouse_block_x + SPEED * x;
			end

			if(y_ov) begin
				mouse_block_y_next <= (y_sign) ? mouse_block_y - SPEED * y : mouse_block_y + SPEED * y;
			end
			else begin
				mouse_block_y_next <= (y_sign) ? mouse_block_y + SPEED * y : mouse_block_y - SPEED * y;
			end
		end
	end
	always @(posedge clk, posedge rst) begin
		if(rst) begin
			mouse_block_x <= 320;
			mouse_block_y <= 240;
		end
		else begin
			mouse_block_x <= mouse_block_x_next;
			mouse_block_y <= mouse_block_y_next;
		end
	end

endmodule
