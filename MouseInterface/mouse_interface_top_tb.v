module MouseInterface_top_tb (
	input wire clk,
    input wire rst,
    inout wire PS2_CLK,          // PS2 Mouse
    inout wire PS2_DATA, 
	input wire sw,
	output wire [2:0] led,
	output wire [6:0] DISPLAY,
	output wire [3:0] DIGIT
);	
	wire mouse_inblock, l_click, cheat_activate;
	wire [4:0] mouse_block_x;
	wire [2:0] mouse_block_y;
	wire [9:0] mouse_x;
	wire [8:0] mouse_y;
	assign led[0] = sw;
	MouseInterface_top mouse_top_inst0(
		.clk(clk),
		.rst(rst),
		.PS2_CLK(PS2_CLK),
		.PS2_DATA(PS2_DATA),
		.mouse_inblock(mouse_inblock),
		.l_click(l_click),
		.cheat_activate(cheat_activate),
		.mouse_x(mouse_x),
		.mouse_y(mouse_y),
		.mouse_block_x(mouse_block_x),
		.mouse_block_y(mouse_block_y)
	);

	wire [3:0] x_hundred = mouse_x / 100;
	wire [3:0] x_ten = (mouse_x / 10) % 10;
	wire [3:0] x_one = mouse_x % 10;
	wire [3:0] y_hundred = mouse_y / 100;
	wire [3:0] y_ten = (mouse_y / 10) % 10;
	wire [3:0] y_one = mouse_y % 10;
	wire [15:0] position; 
	reg [15:0] to_pos;
	SevenSegment SevenSeg_Inst( .clk(clk), .rst(rst), .nums(position), .display(DISPLAY), .digit(DIGIT) );
	// assign postition = (sw) ? {4'd1, x_hundred, x_ten, x_one} : {4'd0, y_hundred, y_ten, y_one};
	assign postition = to_pos;
	always @(*) begin
		if(sw == 1) begin
			to_pos = {4'b0001, x_hundred, x_ten, x_one};
		end else begin
			to_pos = {4'b0000, y_hundred, y_ten, y_one};
		end
	end

	reg [2:0] led_en, led_en_next;
	reg [25:0] counter1, counter2;
	always @(posedge clk, posedge rst) begin
		if(rst) begin
			counter1 <= 0;
		end else if(led_en[0] == 1)begin
			counter1 <= counter1 + 1;
		end	else begin
			counter1 <= 0;
		end
	end

	always @(posedge clk, posedge rst) begin
		if(rst) begin
			counter2 <= 0;
		end else if(led_en[1] == 1)begin
			counter2 <= counter2 + 1;
		end	else begin
			counter2 <= 0;
		end
	end

	always @(*) begin
		led_en_next[0] = led_en[0];
		if(l_click == 1) begin
			led_en_next[0] = 1;
		end else if(led_en[0] == 1 && (1 << 25) == counter1) begin
			led_en_next[0] = 0;
		end
	end

	always @(*) begin
		led_en_next[1] = led_en[1];
		if(cheat_activate == 1) begin
			led_en_next[1] = 1;
		end else if(led_en[1] == 1 && (1 << 25) == counter2) begin
			led_en_next[1] = 0;
		end
	end

endmodule