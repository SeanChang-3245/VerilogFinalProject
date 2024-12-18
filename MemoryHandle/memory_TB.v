module memory_TB();

	reg clk, rst;
	initial begin
		clk = 0;
		forever begin
			#5;
			clk = ~clk;
		end
	end

	reg transmit;
    reg ctrl_en;
    reg ctrl_move_dir;				
    reg [3:0] ctrl_msg_type;			
    reg [4:0] ctrl_block_x;			
    reg [2:0] ctrl_block_y;			
    reg [5:0] ctrl_card;
    reg [2:0] ctrl_sel_len;

    reg interboard_en;
    reg interboard_move_dir;
    reg [3:0] interboard_msg_type;
    reg [4:0] interboard_block_x;
    reg [2:0] interboard_block_y;
    reg [5:0] interboard_card;
    reg [2:0] interboard_sel_len;
endmodule