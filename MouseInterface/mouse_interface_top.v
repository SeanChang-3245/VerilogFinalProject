module MouseInterface_top (
    input wire clk,
    input wire rst,
    input wire interboard_rst,
    inout wire PS2_CLK,          // PS2 Mouse
    inout wire PS2_DATA,         // PS2 Mouse
    
    output wire mouse_inblock,     // 處理滑鼠不在任何一個區塊的情況
    output wire l_click,
    output wire cheat_activate,
    output reg [9:0] mouse_x,		// position on screen
    output reg [8:0] mouse_y,
    output wire [4:0] mouse_block_x,	// mouse in which block
    output wire [2:0] mouse_block_y
);

	wire [7:0] x;
	wire [7:0] y;

	wire x_ov, x_sign, y_ov, y_sign;
	wire o_r_click, o_l_click, valid;

	ps2_mouse mouse_inst_0(
		.i_clk(clk), 
		.i_reset(rst), 
		.i_PS2Clk(PS2_CLK), 
		.i_PS2Data(PS2_DATA),
		.o_x(x), 
		.o_x_ov(x_ov), 
		.o_x_sign(x_sign), 
		.o_y(y), 
		.o_y_ov(y_ov), 
		.o_y_sign(y_sign),
		.o_r_click(o_r_click), 
		.o_l_click(o_l_click), 
		.o_valid(valid)
	);

	parameter NONE = 0;
	parameter RIGHT = 1;
	parameter LEFT = 2;

	parameter SPEED = 1;

	reg [1:0] cur_click;
	reg [2:0] countR, countR_next;
	reg [2:0] countL, countL_next;
	wire [7:0] x_move;
	wire [7:0] y_move;
	reg [9:0] mouse_x_next;
	reg [8:0] mouse_y_next;

	assign cheat_activate = ((cur_click == RIGHT) && (countR == 4)) ? 1 : 0;
	assign l_click = cur_click == LEFT ? 1 : 0;

	// HANDLE CURRENT CLICK	=========================================================================
	always @(*) begin
		cur_click = NONE;
		if(valid) begin
			if(o_r_click) cur_click = RIGHT;
			else if(o_l_click) cur_click = LEFT;
		end
	end

	// CONTINUOUS CLICK COUNTER ======================================================================
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
	// MOUSE MOVEMENT CONTROL ======================================================================
	assign x_move = x_sign ? - (SPEED * x) : SPEED * x;
	assign y_move = y_sign ? SPEED * y : - (SPEED * y);
	always @(*) begin
		mouse_x_next = mouse_x;
		if(valid) begin
			if(!x_ov && (mouse_x + x_move) >= 0 && (mouse_x + x_move) < 640) begin
				mouse_x_next = mouse_x + x_move;
			end else if(x_ov && (mouse_x - x_move) >= 0 && (mouse_x - x_move) < 640) begin
				mouse_x_next = mouse_x - x_move;
			end
		end
	end
	always @(*) begin
		mouse_y_next = mouse_y;
		if(valid) begin
			if(!y_ov && (mouse_y + y_move) >= 0 && (mouse_y + y_move) < 480) begin
				mouse_y_next = mouse_y + y_move;
			end else if(y_ov && (mouse_y - y_move) >= 0 && (mouse_y - y_move) < 480) begin
				mouse_y_next = mouse_y - y_move;
			end
		end
	end
	// STATE REGISTER ===============================================================================
	always @(posedge clk, posedge rst) begin
		if(rst) begin
			countL <= 0;
			countR <= 0;
			mouse_x <= 320;
			mouse_y <= 240;
		end else begin
			countL <= countL_next;
			countR <= countR_next;
			mouse_x <= mouse_x_next;
			mouse_y <= mouse_y_next;
		end
	end

	// ILA debug
	ila_0 ila_0_inst(clk, x, PS2_DATA, x_ov, x_sign, valid);

endmodule
