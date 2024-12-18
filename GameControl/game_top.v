`include "game_macro.v"
`include "../message_macro.v"

module GameControl_top #(
    parameter PLAYER = 0
)(
    input wire clk,
    input wire rst, 
    input wire interboard_rst,
    
    // from user input
    input wire shift_en,                   // from Player_top, decide which operation the player is doing, shift or take/down, not one-pulsef
    input wire start_game,
    input wire rule_valid,
    input wire five_r_click,
    input wire move_left,
    input wire move_right,
    input wire reset_table_raw,
    input wire done_and_next,
    input wire draw_and_next,

    // from memory
    input wire [105:0] available_card,    // used to determine which cards can be drawn
    input wire [8*18*6-1:0] map,

    // from InterboardCommunication
    input wire inter_ready,                // from InterboardCommunication, indicate the module is ready to transmit action done to other side
    input wire interboard_en,             // from InterboardCommunication, one-pulse signal to indicate the transmitted data is valid
    input wire [3:0] interboard_msg_type, // from interboard, used to advance state in FSM
    // input wire [5:0] picked_card,         // from memory, indicate the card that is clicked

    // from MouseInterface
    input wire l_click,
    input wire mouse_inblock,
    input wire [9:0] mouse_x,
    input wire [8:0] mouse_y,
    input wire [4:0] mouse_block_x,       // mouse information
    input wire [2:0] mouse_block_y,       // mouse information

    output wire can_done,                 // indicate the player can finish his turn and switch turn
    output wire can_draw,                 // indicate the player can draw a card and switch turn
    output wire transmit,                 // to InterboardCommunication and Memory, indicate it's this player's turn to transmit data (move)
    output wire ctrl_en,
    output wire ctrl_move_dir,
    output wire [4:0] ctrl_block_x,       // protocol information 
    output wire [2:0] ctrl_block_y,       // protocol information
    output wire [3:0] ctrl_msg_type,
    output wire [5:0] ctrl_card,
    output wire [2:0] ctrl_sel_len,

    output wire [8*18-1:0] sel_card       // to Display, indicate which cards are selected
);
    // Define player constant
    localparam P1 = 0;
    localparam P2 = 1;


    // Message type definition
    localparam TABLE_TAKE = 0;
    localparam TABLE_DOWN = 1;
    localparam TABLE_SHIFT = 2;

    localparam HAND_TAKE = 3;
    localparam HAND_DOWN = 4;
    // localparam HAND_DRAW = 4;

    localparam DECK_DRAW = 5;
    // localparam DECK_DOWN = 6;

    localparam STATE_TURN = 6;
    localparam STATE_RST_TABLE = 7;
    localparam STATE_CHEAT = 8;
    // localparam STATE_RST_GAME = 9;


    // Define states
    localparam INIT = 0;
    
    localparam P1_INIT_DRAW = 1;
    localparam P1_WAIT_IN = 2;
    localparam P1_MOVE = 3;
    localparam P1_SHIFT = 4;

    localparam P2_INIT_DRAW = 5;
    localparam P2_WAIT_IN = 6;
    localparam P2_MOVE = 7;
    localparam P2_SHIFT = 8;

    localparam FIN = 9;
    

    // Define local variable
    reg [3:0] cur_game_state, next_game_state;
    wire valid_card_take, valid_card_down;
    wire [6:0] my_card_cnt;

    // use state to check whether this is my turn
    reg my_turn;
    always @(*) begin
        if(PLAYER == P1) begin
            
        end
        else begin

        end
    end

    // init draw use two states, player one draw 14 cards first, then pass a signal 
    // to player two to notify it is his turn to draw

endmodule

module handle_reset_table#(
    parameter PLAYER = 0
)(
    input wire clk, 
    input wire rst, 
    input wire interboard_rst,

    input wire reset_table_raw,
    input wire [3:0] cur_game_state,
    input wire inter_ready,

    input wire interboard_en,
    input wire [3:0] interboard_msg_type,

    output wire reset_table,

    output wire reset_table_ctrl_en,
    output wire reset_table_ctrl_move_dir,
    output wire [4:0] reset_table_ctrl_block_x,
    output wire [2:0] reset_table_ctrl_block_y,
    output wire [3:0] reset_table_ctrl_msg_type,
    output wire [5:0] reset_table_ctrl_card,
    output wire [2:0] reset_table_ctrl_sel_len
);
endmodule

