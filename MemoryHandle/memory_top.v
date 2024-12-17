module MemoryHandle_top(
    input wire clk,
    input wire rst,
    input wire interboard_rst,

    // from GameControl
    input wire transmit,                    // indicate it's this players turn to move cards, used to select GameControl input or Interbaord Input 
    input wire ctrl_en,
    input wire ctrl_move_dir,				// 0: left, 1: right
    input wire [3:0] ctrl_msg_type,			
    input wire [4:0] ctrl_block_x,			// 0-17
    input wire [2:0] ctrl_block_y,			// 0-7
    input wire [5:0] ctrl_card,
    input wire [2:0] ctrl_sel_len,

    // from InterboardCommunication
    input wire interboard_en,
    input wire interboard_move_dir,
    input wire [3:0] interboard_msg_type,
    input wire [4:0] interboard_block_x,
    input wire [2:0] interboard_block_y,
    input wire [5:0] interboard_card,
    input wire [2:0] interboard_sel_len,

    // to GameControl
    // output reg [5:0] picked_card,           // to GameControl, return the card that is clicked
    output reg [105:0] available_card,      // to GameControl, return which cards can be drawn

    // to Display and RuleCheck
    output reg [6:0] oppo_card_cnt,
    output reg [6:0] deck_card_cnt,
    output reg [8*18*6-1:0] map
    // output reg [8*18-1:0] sel_card,         // to Display, indicate which cards 
);
    localparam TABLE_TAKE = 0;
    localparam TABLE_DOWN = 1;
    localparam TABLE_SHIFT = 2;

    localparam HAND_TAKE = 3;
    localparam HAND_DOWN = 4;

    localparam DECK_DRAW = 5;

    localparam STATE_TURN = 6;
    localparam STATE_RST_TABLE = 7;

    reg en;
    reg move_dir;
    reg [3:0] msg_type;
    reg [4:0] block_x;
    reg [2:0] block_y;
    reg [5:0] card;
    reg [2:0] sel_len;

	wire all_rst, table_rst;
	wire [7:0] position;
	reg [7:0] prev_position;
	reg [7:0] remove_position;
	reg [6:0] oppo_cnt_next, oppo_cnt_cur;
	reg [6:0] deck_cnt_next;

	assign position = block_x + block_y*12;
	assign all_rst = interboard_rst | rst;
	assign table_rst = (en && msg_type == STATE_RST_TABLE);

    // only one of the interboard and game control should be enabled
    always @(*) begin
        if(transmit) begin
            en = ctrl_en;
            move_dir = ctrl_move_dir;
            msg_type = ctrl_msg_type;
            block_x = ctrl_block_x;
            block_y = ctrl_block_y;
            card = ctrl_card;
            sel_len = ctrl_sel_len;
        end
        else begin
            en = interboard_en;
            move_dir = interboard_move_dir;
            msg_type = interboard_msg_type;
            block_x = interboard_block_x;
            block_y = interboard_block_y;
            card = interboard_card;
            sel_len = interboard_sel_len;
        end
    end

    // memory need to store original table before player move cards around, 
    // in order to immediately reset the table after receive message: STATE_RST_TABLE

    // Need to record and calculate cards count by observing command received

    // Need to record the card picked and remove it from its original position after it has been moved
	reg [8*18*6-1:0] map_original;
	reg [8*18*6-1:0] map_next;
	reg [105:0] available_card_next;
	// Calculate Oppo cards ===================================================================================
	always @(*) begin
		oppo_cnt_next = oppo_cnt_cur;
		if(!transmit && en) begin
			if(msg_type == HAND_DOWN) begin
				oppo_cnt_next = oppo_cnt_cur + 1;
			end
			else if(msg_type == HAND_TAKE) begin
				oppo_cnt_next = oppo_cnt_cur - 1;
			end
			else if(msg_type == STATE_RST_TABLE) begin
				oppo_cnt_next = oppo_card_cnt;
			end
		end
	end
	always @(posedge clk) begin
		if(all_rst) begin
			oppo_card_cnt <= 0;
			oppo_cnt_cur <= 0;
		end
		else if(!transmit && en && msg_type == STATE_TURN)begin
			oppo_card_cnt <= oppo_cnt_next;
			oppo_cnt_cur <= oppo_cnt_next;
		end else begin
			oppo_card_cnt <= oppo_card_cnt;
			oppo_cnt_cur <= oppo_cnt_next;
		end
	end
	// DRAW CARD: mark the chosen card as unavailable ==========================================================
	// Also, Calculate Deck Cards
	always @(*) begin
		available_card_next = available_card;
		deck_cnt_next = deck_card_cnt;
		if(en && msg_type == DECK_DRAW && card < 54) begin
			if(available_card[card] == 0 && card < 52) begin
				available_card[card+54] = 0;
			end else begin
				available_card[card] = 0;
			end
			deck_cnt_next = deck_card_cnt - 1;
		end
	end
	// Record Remove Card Position ==============================================================================
	always @(*) begin
		prev_position = remove_position;
		if(transmit && en) begin
			if(msg_type == TABLE_TAKE || msg_type == HAND_TAKE) begin
				prev_position = position;
			end
			else if(msg_type == DECK_DRAW) begin
				prev_position = 144;
			end
		end
		else if(!transmit && en) begin
			if(msg_type == TABLE_TAKE) begin
				prev_position = position;
			end
			else if(msg_type == HAND_TAKE || msg_type == DECK_DRAW) begin
				prev_position = 144;
			end
		end
	end

	// Put Card on Table and Hand ==============================================================================
	integer i;
	always @(*) begin
		map_next = map;
		if(en && msg_type == TABLE_SHIFT) begin
			if(move_dir && (block_x + sel_len -1) < 18) begin
				map_next[position*6+5 -: 6] = 54;
				for(i = position; i < position + sel_len; i = i + 1) begin
					map_next[(i+1)*6+5 -: 6] = map[i*6+5 -: 6];
				end
			end
			else if(!move_dir && (block_x > 0)) begin
				map_next[position*6+5 -: 6] = 54;
				for(i = position; i < position + sel_len; i = i + 1) begin
					map_next[(i-1)*6+5 -: 6] = map[i*6+5 -: 6];
				end
			end
		end
		else if(transmit && en) begin
			if(msg_type == TABLE_DOWN || msg_type == HAND_DOWN) begin
				map_next[position*6+5 -: 6] = card;
				if(remove_position != 144) map_next[remove_position*6+5 -: 6] = 54;
			end
		end
		else if(!transmit && en) begin
			if(msg_type == TABLE_DOWN) begin
				map_next[position*6+5 -: 6] = card;
				if(remove_position != 144) map_next[remove_position*6+5 -: 6] = 54;
			end
			else if(msg_type == HAND_DOWN) begin
				if(remove_position != 144) map_next[remove_position*6+5 -: 6] = 54;
			end
		end
	end

	// State Register ==========================================================================================
	always @(posedge clk) begin
		if(all_rst) begin
			map <= {144{6'd54}};
			map_original <= {144{6'd54}};
		 	available_card <= {106{1'b1}};
			remove_position <= 8'd144;
		end
		else begin
			if(table_rst) begin
				map <= map_original;	// 因為自己手牌不會在對方turn的時候改變，所以在!transmit 的時候分開討論
			end else begin
				map <= map_next;
			end
			if(en && msg_type == STATE_TURN) begin
				map_original <= map_next;
			end else begin
				map_original <= map_original;
			end
			available_card <= available_card_next;
			remove_position <= prev_position;
		end
	end

endmodule