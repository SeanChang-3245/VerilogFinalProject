module MouseInterface_top (
    input wire clk,
    input wire rst,
    input wire interboard_rst,
	input wire [9:0] h_cnt,
	input wire [9:0] v_cnt,
	
    inout wire PS2_CLK,          // PS2 Mouse
    inout wire PS2_DATA,         // PS2 Mouse
    
    output wire mouse_inblock,     // 處理滑鼠不在任何一個區塊的情況
	output wire en_mouse_display,
	output wire [11:0] mouse_pixel,
    output wire l_click,
    output wire cheat_activate,
    output wire [9:0] mouse_x,		// position on screen
    output wire [8:0] mouse_y,
    output wire [4:0] mouse_block_x,	// mouse in which block
    output wire [2:0] mouse_block_y
);

	wire left_click, right_click;
	wire all_rst;
	assign all_rst = rst | interboard_rst;

	wire left_pulse, right_pulse;
	one_pulse op_left_inst( .clk(clk), .pb_db(left_click), .pb_op(left_pulse) );
	one_pulse op_right_inst( .clk(clk), .pb_db(right_click), .pb_op(right_pulse) );

	mouse mouse_inst0(
		.clk(clk), 
		.rst(all_rst),
		.h_cntr_reg(h_cnt), 
		.v_cntr_reg(v_cnt), 
		.PS2_CLK(PS2_CLK), 
		.PS2_DATA(PS2_DATA),
		.enable_mouse_display(en_mouse_display), 
		.MOUSE_X_POS(mouse_x), 
		.MOUSE_Y_POS(mouse_y), 
		.MOUSE_LEFT(left_click), 
		.MOUSE_RIGHT(right_click), 
		.mouse_pixel(mouse_pixel)
	);

	reg [2:0] countR, countR_next;
	reg [2:0] countL, countL_next;

	assign l_click = left_pulse;
	assign cheat_activate = (countR == 5);
	
	always @(*) begin
		countR_next = countR;
		countL_next = countL;
		if(right_pulse) begin
			countR_next = countR + 1;
			countL_next = 0;
		end
		else if(left_pulse) begin
			countR_next = 0;
			countL_next = countL + 1;
		end else if(countR == 5) begin
			countR_next = 0;
		end
	end
	always @(posedge clk) begin
		if(all_rst) begin
			countR <= 0;
			countL <= 0;
		end
		else begin
			countR <= countR_next;
			countL <= countL_next;
		end
	end
endmodule
