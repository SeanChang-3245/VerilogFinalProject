`timescale 1ns/1ns
module a_mod_b_TB();

    reg clk, rst;
    reg [6:0] a, b;
    wire [6:0] ans;
    wire done, ready;
    reg start;

    a_mod_b amb(.clk(clk), .rst(rst), .interboard_rst(0), .a(a), .b(b), .start(start),
                .ready(ready), .done(done), .ans(ans));

    initial begin
        clk = 0;
        forever begin
            #5;
            clk = ~clk;
        end
    end


    initial begin
        #500;
        $finish();
    end

    initial begin
        #10;
        rst = 1;
        #10;
        rst = 0;

        #10;
        a = 37;
        b = 6;
        start = 1;
        #10;
        b = 10;
        start = 0;

        #150;

        #10;
        a = 35;
        b = 7;
        start = 1;
        #10;
        start = 0;

    end

    always@* begin
        if(done) begin
            $display("The answer is %d", ans);
        end
    end

endmodule
