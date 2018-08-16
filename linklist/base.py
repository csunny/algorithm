#!/usr/bin/env python3

"""
This document is created by magic at 2018/8/16
"""


class LinkNode:
    """
    doc node
    """

    def __init__(self, item, next=None):
        self._item = item
        self._next = next

    @property
    def item(self):
        return self._item

    @item.setter
    def item(self, item):
        self._item = item

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, next):
        self._next = next


def test_link_node():
    node_one = LinkNode(1)

    node_two = LinkNode(2, node_one)

    print(node_two.item)

    node_two.item = 100
    print(node_two.item)


class LinkedList:

    def __init__(self, head=None):
        if head:
            self.head = head
        else:
            self.head = LinkNode(0)

    def add(self, item):
        """
        添加node  头部插入的方式
        :param item:
        :return:
        """
        new_node = LinkNode(item)
        new_node.next = self.head
        self.head = new_node

    def add_tail(self, item):
        """
        尾部插入的方式
        :param item:
        :return:
        """
        new_node = LinkNode(item)

        point = self.head
        while point.next:
            point = point.next
        point.next = new_node

    def reverse(self):
        """
        反转链表
        :return:
        """
        if not self.head or not self.head.next:
            return self.head

        reversed_head = self.head
        head = self.head.next

        reversed_head.next = None

        p = head.next
        while head:
            head.next = reversed_head
            reversed_head = head

            head = p
            if p:
                p = p.next
        self.head = reversed_head

    def reverse_recursion(self):
        """
        递归的方式反转链表
        :return:
        """
        pass

    def remove(self, index):
        """
        根据索引删除链表中的元素
        :param index:
        :return:
        """
        point = self.head
        for _ in range(index):
            point = point.next

        point.next = point.next.next

    def insert(self, index, item):
        point = self.head
        for _ in range(index):
            point = point.next

        new_node = LinkNode(item)
        new_node.next = point.next.next
        point.next = new_node

    def __len__(self):
        length = 1

        point = self.head
        while point.next:
            length += 1
            point = point.next
        return length


def build_link_list():
    lined_list = LinkedList()
    lined_list.add_tail("magic 1")
    lined_list.add_tail("tom 2")
    lined_list.add_tail("ju 3")
    lined_list.add_tail("mark 4")
    lined_list.add_tail("s 5")
    lined_list.add_tail(6)
    lined_list.add_tail(7)

    return lined_list


def test():
    link_list = build_link_list()
    print(len(link_list))

    # link_list.reverse()
    # link_list.remove(2)
    # link_list.insert(3, "I am insert 3")
    head = link_list.head
    while head:
        print(head.item)
        head = head.next


if __name__ == '__main__':
    # test_link_node()
    test()
