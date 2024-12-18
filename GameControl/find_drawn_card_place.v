module find_draw_card_place(
    input wire clk,
    input wire rst, 
    input wire interboard_rst,
    input wire [8*18*6-1:0] map,

    output reg [5:0] card_place // 0~35
);
    localparam NO_CARD = 54;
    integer i;

    always@* begin
        for(i = 0; i <= 35; i = i+1) begin
            if(map[8*18*6-1 - i*6 -: 6] == NO_CARD) begin
                card_place = i;
            end
        end
    end
endmodule