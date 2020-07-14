#define PARAM_AMOUNT 7
#define CANDLE_DATA_WIDTH 5

// extern "C++"
// {
//     template <typename T>
//     class List
//     {
//     private:
//         class Node
//         {
//         public:
//             T value;
//             Node *next = nullptr, *previous = nullptr;
        
//             __device__
//             Node(T &value) : value(value) {};
        
//         };

//     public:
//         class Iterator
//         {
//             friend List;
//         private:
//             List<T>::Node *currentItem;
//         public:
//             __device__
//             Iterator(List<T>::Node *item) : currentItem(item) {}
            
//             __device__
//             T& operator* ()
//             {
//                 return this->currentItem->value;
//             }

//             __device__
//             bool last()
//             {
//                 return this->currentItem == nullptr;
//             }

//             __device__
//             List<T>::Iterator operator++(int)
//             {
//                 this->currentItem = this->currentItem->next;
//                 return *this;
//             }
//         };

//     private:
//         List<T>::Node *first = nullptr, *last = nullptr;
//     public:
//         __device__
//         void push(T value) 
//         {
//             List<T>::Node* node = new List<T>::Node(value);
//             if (this->last == nullptr)
//             {
//                 this->last = node;
//                 this->first = node;
//             }
//             else 
//             {
//                 this->last->next = node;
//                 node->previous = this->last;
//                 this->last = node;
//             }
//         }

//         __device__
//         List<T>::Iterator begin() 
//         {
//             return List<T>::Iterator(this->first);
//         }

//         __device__
//         ~List() 
//         {
//             if (this->first == nullptr) return;

//             int i = 0;

//             List<T>::Node *currentNode = this->first;
//             while (currentNode->next != nullptr)
//             {
//                 i++;
//                 delete currentNode;
//                 currentNode = currentNode->next;
//             }
//             i++;
//             delete currentNode;
//             printf("%i ", i);
//         }

//     };
// }

class Order {
public:
    double price;
    int volume;

    __device__ 
    Order(double price = 0, int volume = 0) : price(price), volume(volume) {}

    __device__ 
    bool isFilled(double high, double low) {
        return this->price >= low && this->price <= high;
    }

    __device__ 
    int getVolume() {return this->volume;}

    __device__ 
    double getPrice() {return this->price;}
};

class GridTradingSimulator {
public:
    struct Parameters {
        __int64 orderPairs;
        __int64 orderStartSize;
        __int64 orderStepSize;
        double interval;
        double minSpread;
        __int64 minPosition;
        __int64 maxPosition;
    };

    struct PositionData {
        int volume = 0;
        double price = 0.f;
    };

private:
    __int64 *candleData;
    __int64 candleDataSize;
    GridTradingSimulator::Parameters params;
    GridTradingSimulator::PositionData position;

    double feePercentage = -0.000225;
    double balanceBTC = 1;

public:
    __device__ 
    GridTradingSimulator(
        __int64 *candleData, 
        __int64 candleDataSize, 
        GridTradingSimulator::Parameters params
    ) : 
        candleData(candleData), 
        candleDataSize(candleDataSize), 
        params(params)
        {}

    __device__
    void onOrderFilled(Order &order)
    {
        if (position.volume == 0) 
        {
            // printf("1 ");
            position.volume = order.volume;
            position.price = order.price;
        }
        else if (position.volume < 0 && order.volume < 0 || position.volume > 0 && order.volume > 0)
        {
            // printf("2 ");
            position.price = order.price * order.volume + position.price * position.volume;
            position.volume += order.volume;
            position.price /= position.volume;
        }
        else if (abs(position.volume) >= abs(order.volume))
        {
            // printf("3 ");
            position.volume += order.volume;
            this->balanceBTC += order.volume * (-1.f / position.price + 1.f / order.price);
        }
        else 
        {
            // printf("4 ");
            this->balanceBTC += order.volume * (1.f / position.price - 1.f / order.price);
            position.volume += order.volume;
            position.price = order.price;
        }

        double fee = this->feePercentage * order.volume / order.price;
        this->balanceBTC -= fee;
    }

    __device__
    bool isBankrupt(double price)
    {
        return this->balanceBTC - position.volume * (1.f / position.price - 1.f / price) <= 0.f;
    }

    __device__ 
    __int32 getResult(){
        for (int i = 0; i < this->candleDataSize; ++i)
        {
            double open = double(this->candleData[i * CANDLE_DATA_WIDTH + 0]) / 10000.f;
            double high = double(this->candleData[i * CANDLE_DATA_WIDTH + 1]) / 10000.f;
            double low = double(this->candleData[i * CANDLE_DATA_WIDTH + 2]) / 10000.f;

            // List<Order> currentOrders = this->makeOrders(open);
            double interval = open * this->params.interval;
            double minSpread = open * this->params.minSpread;

            Order *orders = new Order[this->params.orderPairs * 2];

            for (int i = 0; i < this->params.orderPairs; i++)
            {
                double price;
                int volume;

                if (this->position.volume < this->params.maxPosition){
                    price = open + minSpread / 2 + i * interval;
                    price = double(int(price * 2)) / 2.f;
                    volume = this->params.orderStartSize + this->params.orderStepSize * i;
                    orders[i*2] = Order(price, volume);
                }

                if ( this->position.volume > this->params.minPosition){
                    price = open - minSpread / 2 - i * interval;
                    price = double(int(price * 2)) / 2.f;
                    volume = -(this->params.orderStartSize + this->params.orderStepSize * i);
                    orders[i*2 + 1] = Order(price, volume);
                }
            }
            
            for (int i = 0; i < this->params.orderPairs * 2; ++i) 
            {
                Order currentOrder = orders[i];
                if (currentOrder.volume != 0 && currentOrder.isFilled(high, low))
                {
                    this->onOrderFilled(currentOrder);
                }
            }

            delete[] orders;

            if (this->isBankrupt(high) || this->isBankrupt(low))
            {
                return 0;
            }
        }


        return int(this->balanceBTC * 10000000);
    }
};

__global__ 
void start(__int32 *result, __int64 *params, __int64 *candleData, int *candleDataSize)
{
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    //int threadId = blockId * blockDim.x + threadIdx.x;

    // printf("%i ", blockId);

    GridTradingSimulator::Parameters simulationParameters = {
        params[blockId*PARAM_AMOUNT    ],
        params[blockId*PARAM_AMOUNT + 1],
        params[blockId*PARAM_AMOUNT + 2],
        double(params[blockId*PARAM_AMOUNT + 3]) / 1000000.f,
        double(params[blockId*PARAM_AMOUNT + 4]) / 1000000.f,
        params[blockId*PARAM_AMOUNT + 5],
        params[blockId*PARAM_AMOUNT + 6]
    };

    GridTradingSimulator sim = GridTradingSimulator(candleData, *candleDataSize, simulationParameters);


    result[blockId] = sim.getResult();
    // printf("%i ", blockId);
}