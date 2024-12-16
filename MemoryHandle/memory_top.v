module MemoryHandle_top(
    input wire clk,
    input wire rst,
    input wire interboard_rst,

    // from GameControl
    input wire transmit,                    // indicate it's this players turn to move cards, used to select GameControl input or Interbaord Input 
    input wire ctrl_en,
    input wire ctrl_move_dir,
    input wire [3:0] ctrl_msg_type,
    input wire [4:0] ctrl_block_x,
    input wire [2:0] ctrl_block_y,
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
    reg en;
    reg move_dir;
    reg [3:0] msg_type;
    reg [4:0] block_x;
    reg [2:0] block_y;
    reg [5:0] card;
    reg [2:0] sel_len;

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

endmodule