module MemoryHandle_top(
    input wire clk,
    input wire rst,
    
    // from GameControl
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
    output reg [5:0] picked_card,           // to GameControl, return the card that is clicked
    output reg [105:0] available_card,      // to GameControl, return which cards can be drawn

    // to Display and RuleCheck
    output reg [8*18*6-1:0] map
    // output reg [8*18-1:0] sel_card,         // to Display, indicate which cards 
);
    wire en;
    wire move_dir;
    wire [3:0] msg_type;
    wire [4:0] block_x;
    wire [2:0] block_y;
    wire [5:0] card;
    wire [2:0] sel_len;

    // only one of the interboard and game control should be enabled
    always @(*) begin
        if(ctrl_en && !interboard_en) begin
            
        end
        else if(!ctrl_en && interboard_en) begin

        end
        else begin

        end
    end


endmodule